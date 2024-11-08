from flask import Flask, request, render_template, jsonify, send_from_directory
import cv2
import numpy as np
import torch
import open3d as o3d
import os
import time
from transformers import AutoImageProcessor, AutoModelForDepthEstimation
from PIL import Image

# 設置環境變量以啟用 CPU 回退
os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'

app = Flask(__name__)

# 修改模型名稱部分
model_name = "depth-anything/Depth-Anything-V2-Large-hf"

# 加載模型時添加 token
auth_token = "hf_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"  # 替換為您的 token

# 確保使用 PyTorch
print("PyTorch version:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())

try:
    processor = AutoImageProcessor.from_pretrained(model_name, token=auth_token)
    model = AutoModelForDepthEstimation.from_pretrained(model_name, token=auth_token)
except Exception as e:
    print(f"Error loading model: {e}")
    raise

# 修改設備選擇邏輯
if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")  # 直接使用 CPU
print(f"Using device: {device}")
model.to(device)

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
    try:
        # 直接處理圖像
        inputs = processor(images=img, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}

        # 使用模型進行預測
        with torch.no_grad():
            outputs = model(**inputs)
            predicted_depth = outputs.predicted_depth

        # 處理深度圖
        depth_map = torch.nn.functional.interpolate(
            predicted_depth.unsqueeze(1),
            size=img.shape[:2],
            mode="bicubic",
            align_corners=False,
        ).squeeze().cpu().numpy()
        
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
    # 直接使用深度圖，不需要對數映射
    depth = (depth_map * 0.3).astype(np.float32)  # 縮小深度範圍
    
    # 使用中值濾波器來減少噪點，使用較小的核心大小
    depth = cv2.medianBlur(depth, 3)
    
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
        depth_scale=500,     # 調整深度縮放
        depth_trunc=5.0,     # 減小深度截斷閾值
        stride=1            # 增加步長以減少點數
    )
    
    # 為點雲添加顏色
    color = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    color = cv2.resize(color, (width, height))
    
    # 創建與點雲點數相同的顏色數組
    points = np.asarray(pcd.points)
    colors = np.zeros((len(points), 3))
    
    # 對每個點進行反投影，找到對應的圖像像素
    for i, point in enumerate(points):
        # 計算點在圖像中的投影位置
        x = int((point[0] * fx / point[2]) + cx)
        y = int((point[1] * fy / point[2]) + cy)
        
        # 確保坐標在有效範圍內
        if 0 <= x < width and 0 <= y < height:
            colors[i] = color[y, x] / 255.0
    
    # 設置點雲顏色
    pcd.colors = o3d.utility.Vector3dVector(colors)
    
    # 使用較輕量的統計濾波器移除離群點
    pcd, _ = pcd.remove_statistical_outlier(
        nb_neighbors=20,     # 減少鄰居點數
        std_ratio=2.0        # 放寬標準差比率
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