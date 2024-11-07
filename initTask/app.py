from flask import Flask, request, render_template, jsonify, send_from_directory
import cv2
import numpy as np
import torch
import open3d as o3d
import os
import time

app = Flask(__name__)

# 加載 MiDaS 模型
model_type = "DPT_Large"  # 使用更高精度的模型
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
    
    # 將圖像轉換為點雲
    point_cloud = process_image_to_point_cloud(img)
    
    # 使用時間戳生成唯一文件名
    timestamp = int(time.time())
    point_cloud_file = os.path.join(OUTPUT_FOLDER, f'point_cloud_{timestamp}.ply')
    save_point_cloud(point_cloud, point_cloud_file)
    
    return jsonify({"message": "3D模型已生成", "file_path": f"/{point_cloud_file}"})

@app.route('/static/output/<path:filename>')
def download_file(filename):
    return send_from_directory(OUTPUT_FOLDER, filename)

def process_image_to_point_cloud(img):
    # 將 OpenCV 圖像轉換為 RGB
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # 應用轉換
    input_batch = transform(img_rgb).unsqueeze(0)  # 確保是四維張量

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

    # 將深度圖轉換為點雲
    point_cloud = depth_map_to_point_cloud(depth_map, img_rgb)

    return point_cloud

def depth_map_to_point_cloud(depth_map, img_rgb):
    h, w = depth_map.shape
    fx = fy = 1.0  # 假設焦距為1，這裡可以根據實際情況調整
    cx, cy = w / 2, h / 2

    # 創建點雲
    points = []
    colors = []

    for v in range(h):
        for u in range(w):
            z = depth_map[v, u]
            if z > 0:  # 忽略無效深度
                x = (u - cx) * z / fx
                y = (v - cy) * z / fy
                points.append((x, y, z))
                colors.append(img_rgb[v, u] / 255.0)

    return np.array(points), np.array(colors)

def save_point_cloud(point_cloud, filename):
    points, colors = point_cloud
    # 使用 Open3D 保存點雲
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(colors)
    o3d.io.write_point_cloud(filename, pcd)

if __name__ == '__main__':
    app.run(debug=True)