import numpy as np
import torch
import os
import json
import smplx
import matplotlib.pyplot as plt
from pytorch3d.renderer import look_at_view_transform, FoVPerspectiveCameras

# ==========================================
# 0. 环境补丁 (必须在加载 smplx 之前运行)
# ==========================================
def patch_numpy():
    # 修复 NumPy 1.24+ 移除 np.bool 等别名导致 chumpy 报错的问题
    names = {
        'bool': bool, 'int': int, 'float': float, 'complex': complex,
        'object': object, 'str': str, 'unicode': str,
        'nan': float('nan'), 'inf': float('inf')
    }
    for name, obj in names.items():
        if not hasattr(np, name):
            setattr(np, name, obj)

patch_numpy()

# ==========================================
# 1. 基础配置
# ==========================================
device = torch.device("cuda:0")
MODEL_DIR = "/root/smpl_models_extracted/SMPL_python_v.1.1.0/smpl/models"
JSON_DIR  = "/root/autodl-tmp/openpose_results"

# 映射关系 (OpenPose 25点 -> SMPL 24点)
JOINT_MAP = {0: 8, 1: 12, 2: 9, 4: 13, 5: 10, 7: 14, 8: 11, 12: 1, 15: 0, 16: 5, 17: 2, 18: 6, 19: 3, 20: 7, 21: 4}
smpl_indices = torch.tensor(list(JOINT_MAP.keys())).long().to(device)
op_indices   = torch.tensor(list(JOINT_MAP.values())).long().to(device)

# ==========================================
# 2. 加载模型与数据
# ==========================================
print("正在加载 SMPL 模型...")
smpl = smplx.create(MODEL_DIR, model_type='smpl', gender='male', batch_size=1).to(device)

print("正在读取 OpenPose JSON (View 0)...")
path = os.path.join(JSON_DIR, "view_0_img_keypoints.json")
if not os.path.exists(path):
    print(f"错误：找不到文件 {path}")
    exit()

with open(path, "r") as f:
    data = json.load(f)
    if not data["people"]:
        print("错误：JSON 中没有检测到人")
        exit()
    p = np.array(data["people"][0]["pose_keypoints_2d"]).reshape(-1, 3)
    gt_2d_view0 = torch.from_numpy(p[:, :2]).float().to(device)

# ==========================================
# 3. 执行诊断绘图
# ==========================================
print("--- 启动坐标对齐诊断 ---")
with torch.no_grad():
    # A. 先定义（创建）初始的 transl
    # 我们先给它一个基础值 [0, 2.5, 0]
    transl = torch.tensor([[0, 2.5, 0]], device=device)
    
    # B. 修改（修正）它
    # 根据之前的诊断，我们需要把它往上提
    transl[:, 1] += 0.85 # 核心修正：把模型往上提
    transl[:, 0] += 0.1 # 横向微调
    
    # C. 生成模型输出（这步会产生 out 和 joints_3d）
    out = smpl(
        body_pose=torch.zeros((1, 69), device=device),
       global_orient = torch.tensor([[0, 0, 0]], device=device),
        transl=transl
    )
    joints_3d = out.joints[:, smpl_indices]
    
    # D. 投影到 1024x1024 屏幕
    # 注意：这里的 at=((0, 2.5, 0),) 是相机的注视点，要和模型位置匹配
    R, T = look_at_view_transform(dist=3.8, elev=0, azim=0, at=((0, 2.5, 0),))    
    cam = FoVPerspectiveCameras(device=device, R=R, T=T)
    proj = cam.transform_points_screen(joints_3d, image_size=((1024, 1024),))
    proj_2d = proj[0, ..., :2].cpu().numpy()
    
    gt_2d = gt_2d_view0[op_indices].cpu().numpy()
    
    # E. 绘图
    plt.figure(figsize=(10, 10))
    plt.scatter(gt_2d[:, 0], gt_2d[:, 1], c='blue', s=50, label='OpenPose GT (Blue)')
    plt.scatter(proj_2d[:, 0], proj_2d[:, 1], c='red', s=50, label='SMPL Initial (Red)')
    
    for i in range(len(gt_2d)):
        plt.plot([gt_2d[i, 0], proj_2d[i, 0]], [gt_2d[i, 1], proj_2d[i, 1]], 'gray', alpha=0.3)
        plt.text(gt_2d[i, 0], gt_2d[i, 1], str(i), color='blue', fontsize=9)

    plt.gca().invert_yaxis()
    plt.legend()
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.title("After Manual Fix: Check if Red and Blue overlap better now.")
    plt.savefig("diagnostic_plot_fixed.png")

print("\n--- 诊断完成 ---")
print(f"修正后的诊断图已保存至: {os.getcwd()}/diagnostic_plot_fixed.png")
print(f"OpenPose 中心点: {gt_2d.mean(0)}")
print(f"SMPL 投影中心点: {proj_2d.mean(0)}")
print(f"平均像素距离 (误差): {np.linalg.norm(gt_2d - proj_2d, axis=1).mean():.2f}")