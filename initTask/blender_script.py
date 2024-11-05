import bpy
import sys

def create_3d_model_from_point_cloud(point_cloud_file):
    # 清除所有物件
    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.delete(use_global=False)

    # 確保 PLY 匯入插件已啟用
    try:
        bpy.ops.wm.ply_import(filepath=point_cloud_file)
    except AttributeError:
        print("PLY import operator not found. Please ensure the PLY import add-on is enabled.")
        return

    # 這裡可以加入更多的3D模型處理邏輯

if __name__ == "__main__":
    argv = sys.argv
    point_cloud_file = argv[argv.index("--") + 1]
    create_3d_model_from_point_cloud(point_cloud_file) 