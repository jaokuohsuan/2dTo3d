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
    
    # 將深度圖轉換為點雲
    point_cloud = depth_map_to_point_cloud(depth_map, img)
    
    # 使用時間戳生成唯一文件名
    timestamp = int(time.time())
    point_cloud_file = os.path.join(OUTPUT_FOLDER, f'point_cloud_{timestamp}.ply')
    o3d.io.write_point_cloud(point_cloud_file, point_cloud)
    
    return jsonify({"message": "3D模型已生成", "file_path": f"/{point_cloud_file}"})

@app.route('/static/output/<path:filename>')
def download_file(filename):
    return send_from_directory(OUTPUT_FOLDER, filename)

def generate_depth_map(img):
    # 將 OpenCV 圖像轉換為 RGB
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    try:
        # 應用轉換
        input_batch = transform(img_rgb)
        
        # 確保輸入是 4D 張量 (batch_size, channels, height, width)
        if len(input_batch.shape) == 3:
            input_batch = input_batch.unsqueeze(0)
        
        # 確認是否可用 GPU
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        midas.to(device)
        input_batch = input_batch.to(device)
        
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
        
        # 保存深度圖以供查看
        depth_visualization = (depth_map * 255).astype(np.uint8)
        timestamp = int(time.time())
        cv2.imwrite(os.path.join(OUTPUT_FOLDER, f'depth_map_{timestamp}.png'), depth_visualization)
        
        return depth_map
        
    except Exception as e:
        print("Error in generate_depth_map:", str(e))
        raise

def depth_map_to_point_cloud(depth_map, img):
    # 調整深度映射，使用對數映射來壓縮遠處的深度值
    depth = depth_map.copy()
    
    # 對數映射，壓縮遠處的深度值
    depth = np.log(depth + 1) / np.log(2)  # log2(depth + 1)
    
    # 再次正規化到合適的範圍
    depth = (depth * 0.5).astype(np.float32)  # 縮小深度範圍
    depth_image = o3d.geometry.Image(depth)
    
    # 設置相機內參
    width = depth_map.shape[1]
    height = depth_map.shape[0]
    fx = width
    fy = width
    cx = width / 2
    cy = height / 2
    
    intrinsic = o3d.camera.PinholeCameraIntrinsic(
        width, height, fx, fy, cx, cy
    )
    
    # 從深度圖創建點雲
    pcd = o3d.geometry.PointCloud.create_from_depth_image(
        depth_image,
        intrinsic,
        depth_scale=0.5,    # 調整深度縮放
        depth_trunc=1.0,    # 調整深度截斷閾值
        stride=1
    )
    
    # 為點雲添加顏色
    color = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    color = cv2.resize(color, (width, height))
    
    # 根據深度圖的有效點為點雲著色
    colors = []
    for i in range(height):
        for j in range(width):
            if depth[i, j] > 0:
                colors.append(color[i, j] / 255.0)
    
    if len(colors) > 0:
        pcd.colors = o3d.utility.Vector3dVector(np.array(colors))
    
    # 移除離群點
    pcd, _ = pcd.remove_statistical_outlier(
        nb_neighbors=30,
        std_ratio=2.0  # 調整標準差比率
    )
    
    # 修正旋轉矩陣，解決左右鏡像問題
    R = np.array([
        [-1, 0, 0],  # 改變 x 軸方向
        [0, -1, 0],
        [0, 0, -1]
    ])
    pcd.rotate(R, center=(0, 0, 0))
    
    return pcd

if __name__ == '__main__':
    app.run(debug=True)