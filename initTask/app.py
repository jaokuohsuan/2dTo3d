from flask import Flask, request, render_template, jsonify, send_from_directory
import cv2
import numpy as np
import torch
import open3d as o3d
import os
import time

app = Flask(__name__)

# 加載 MiDaS 模型
model_type = "DPT_Large"
midas = torch.hub.load("intel-isl/MiDaS", model_type)
midas.eval()

# 設定轉換
transform = torch.hub.load("intel-isl/MiDaS", "transforms").dpt_transform

# 設定靜態文件夾
OUTPUT_FOLDER = 'static/output'
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload():
    file = request.files['file']
    if not file:
        return jsonify({"error": "No file uploaded"}), 400

    # 讀取圖像
    img = cv2.imdecode(np.frombuffer(file.read(), np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        return jsonify({"error": "Invalid image format"}), 400
    
    # 生成深度圖
    depth_map = generate_depth_map(img)
    
    # 將深度圖和原始圖像轉換為點雲
    point_cloud = depth_map_to_point_cloud(depth_map, img)
    
    # 使用時間戳生成唯一文件名
    timestamp = int(time.time())
    point_cloud_file = os.path.join(OUTPUT_FOLDER, f'point_cloud_{timestamp}.ply')
    save_point_cloud(point_cloud, point_cloud_file)
    
    return jsonify({"message": "3D模型已生成", "file_path": f"/{point_cloud_file}"})

@app.route('/static/output/<path:filename>')
def download_file(filename):
    return send_from_directory(OUTPUT_FOLDER, filename)

def generate_depth_map(img):
    # 將 OpenCV 圖像轉換為 RGB
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # 調整圖像大小以提高處理效率
    height, width = img_rgb.shape[:2]
    target_width = 384
    target_height = int(height * (target_width / width))
    img_resized = cv2.resize(img_rgb, (target_width, target_height))

    # 應用轉換
    input_batch = transform(img_resized).unsqueeze(0)

    # 確認是否可用 GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    midas.to(device)
    input_batch = input_batch.to(device)

    # 檢查輸入張量的形狀
    if len(input_batch.shape) == 5:
        input_batch = input_batch.squeeze(1)  # 去掉多餘的維度

    if len(input_batch.shape) != 4:
        raise ValueError(f"Input tensor does not have 4 dimensions as expected, got {input_batch.shape}")

    # 生成深度圖
    with torch.no_grad():
        prediction = midas(input_batch)
        prediction = torch.nn.functional.interpolate(
            prediction.unsqueeze(1),
            size=img_rgb.shape[:2],
            mode="bicubic",
            align_corners=False
        ).squeeze()

    depth_map = prediction.cpu().numpy()
    
    # 正規化深度圖到 0-1 範圍
    depth_min = depth_map.min()
    depth_max = depth_map.max()
    depth_map = (depth_map - depth_min) / (depth_max - depth_min)
    
    # 應用高斯模糊以減少噪點
    depth_map = cv2.GaussianBlur(depth_map, (5, 5), 0)
    
    return depth_map

def depth_map_to_point_cloud(depth_map, img):
    h, w = depth_map.shape
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # 設置相機參數
    fx = 500  # 焦距
    fy = 500
    cx = w / 2  # 光學中心
    cy = h / 2
    
    # 創建網格點
    y, x = np.mgrid[0:h, 0:w]
    
    # 計算3D點，翻轉 y 軸方向
    z = depth_map * 1000  # 縮放深度值
    x = (x - cx) * z / fx
    y = -(y - cy) * z / fy  # 加上負號來翻轉 y 軸
    
    # 創建點雲
    mask = z > 0  # 只保留有效深度的點
    points = np.stack((x[mask], y[mask], z[mask]), axis=-1)
    colors = img_rgb[mask] / 255.0
    
    # 降採樣以減少點的數量，但保持較高的採樣率
    sample_rate = 0.6  # 增加採樣率到 30%
    indices = np.random.choice(
        points.shape[0], 
        size=int(points.shape[0] * sample_rate), 
        replace=False
    )
    points = points[indices]
    colors = colors[indices]
    
    # 根據深度值過濾離群點
    z_mean = np.mean(points[:, 2])
    z_std = np.std(points[:, 2])
    z_mask = np.abs(points[:, 2] - z_mean) < 2 * z_std
    points = points[z_mask]
    colors = colors[z_mask]
    
    return points, colors

def save_point_cloud(point_cloud, filename):
    points, colors = point_cloud
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(colors)
    
    # 調整離群點移除的參數
    pcd, _ = pcd.remove_statistical_outlier(
        nb_neighbors=30,  # 增加鄰居點數
        std_ratio=2.0
    )
    
    # 可選：進行法向量估計以改善視覺效果
    pcd.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(
            radius=0.1, 
            max_nn=30
        )
    )
    
    # 保存點雲
    o3d.io.write_point_cloud(filename, pcd)

if __name__ == '__main__':
    app.run(debug=True)