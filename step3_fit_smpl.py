import numpy as np
import torch
import os
import json
import smplx
import matplotlib.pyplot as plt
from pytorch3d.io import save_obj
from pytorch3d.renderer import look_at_view_transform, FoVPerspectiveCameras

# 0. 环境补丁
def patch_numpy():
    names = {'bool': bool, 'int': int, 'float': float, 'complex': complex,
             'object': object, 'str': str, 'unicode': str,
             'nan': float('nan'), 'inf': float('inf')}
    for name, obj in names.items():
        if not hasattr(np, name): setattr(np, name, obj)
patch_numpy()

device = torch.device("cuda:0")

# 1. 路径配置
MODEL_DIR = "/root/smpl_models_extracted/SMPL_python_v.1.1.0/smpl/models"
JSON_DIR  = "/root/autodl-tmp/openpose_results"
SAVE_PATH = "/root/autodl-tmp/final_fitted_smpl.obj"
VIS_PATH_STAGE1 = "/root/autodl-tmp/alignment_stage1_check.png"
VIS_PATH_FINAL = "/root/autodl-tmp/alignment_final_check.png"

# 2. 映射关系
JOINT_MAP = {0: 8, 1: 12, 2: 9, 4: 13, 5: 10, 7: 14, 8: 11, 12: 1, 15: 0, 16: 5, 17: 2, 18: 6, 19: 3, 20: 7, 21: 4}
smpl_indices = torch.tensor(list(JOINT_MAP.keys())).long().to(device)
op_indices   = torch.tensor(list(JOINT_MAP.values())).long().to(device)

# 3. 加载数据
smpl = smplx.create(MODEL_DIR, model_type='smpl', gender='male', batch_size=1).to(device)
gt_joints_2d, joint_conf = [], []
for i in range(8):
    path = os.path.join(JSON_DIR, f"view_{i}_img_keypoints.json")
    with open(path, "r") as f:
        data = json.load(f)
        p = np.array(data["people"][0]["pose_keypoints_2d"]).reshape(-1, 3) if data["people"] else np.zeros((25, 3))
        gt_joints_2d.append(torch.from_numpy(p[:, :2]).float())
        joint_conf.append(torch.from_numpy(p[:, 2]).float())
gt_joints_2d = torch.stack(gt_joints_2d).to(device)
joint_conf = torch.stack(joint_conf).to(device)

# 4. 初始化
global_orient = torch.tensor([[0, 0, 0]], device=device, dtype=torch.float32, requires_grad=True)
transl = torch.tensor([[0.15, 3.25, 0.0]], device=device, dtype=torch.float32, requires_grad=True)
body_pose = torch.zeros((1, 69), device=device, requires_grad=True)
betas = torch.zeros((1, 10), device=device)

def plot_debug(smpl_out, title, save_path):
    with torch.no_grad():
        j3d = smpl_out.joints[:, smpl_indices]
        fig = plt.figure(figsize=(20, 10))
        for i in range(8):
            R, T = look_at_view_transform(dist=4.5, elev=0, azim=i * 45, at=((0, 2.5, 0),))
            cam = FoVPerspectiveCameras(device=device, R=R, T=T, fov=40)
            proj = cam.transform_points_screen(j3d, image_size=((1024, 1024),))[0, ..., :2].cpu().numpy()
            gt = gt_joints_2d[i, op_indices].cpu().numpy()
            ax = fig.add_subplot(2, 4, i+1)
            ax.scatter(gt[:, 0], gt[:, 1], c='blue', s=10)
            ax.scatter(proj[:, 0], proj[:, 1], c='red', s=20, marker='x')
            for j in range(len(gt)): ax.plot([gt[j, 0], proj[j, 0]], [gt[j, 1], proj[j, 1]], 'gray', alpha=0.2)
            ax.invert_yaxis(); ax.set_title(f"View {i}")
        plt.savefig(save_path); plt.close()

# 5. Stage 1: 重心对齐
print("--- 开始优化 ---")
print("Stage 1: 对齐中心位移...")
opt1 = torch.optim.Adam([global_orient, transl], lr=0.02)
for _ in range(200):
    opt1.zero_grad()
    out = smpl(body_pose=torch.zeros_like(body_pose), global_orient=global_orient, transl=transl)
    loss = 0
    for i in range(8):
        R, T = look_at_view_transform(dist=4.5, elev=0, azim=i * 45, at=((0, 2.5, 0),))
        cam = FoVPerspectiveCameras(device=device, R=R, T=T)
        proj = cam.transform_points_screen(out.joints[:, smpl_indices], image_size=((1024, 1024),))[..., :2]
        loss += torch.nn.functional.mse_loss(proj[0].mean(0), gt_joints_2d[i, op_indices].mean(0))
    loss.backward(); opt1.step()

# 6. Stage 2: 肢体拟合 (1000次高精度迭代)
print("Stage 2: 拟合肢体姿态...")
opt2 = torch.optim.Adam([body_pose, global_orient, transl], lr=0.005)
for step in range(1001):
    opt2.zero_grad()
    out = smpl(body_pose=body_pose, global_orient=global_orient, transl=transl)
    joints = out.joints[:, smpl_indices]
    loss_2d = 0
    for i in range(8):
        R, T = look_at_view_transform(dist=4.5, elev=0, azim=i * 45, at=((0, 2.5, 0),))
        cam = FoVPerspectiveCameras(device=device, R=R, T=T, fov=40)
        proj = cam.transform_points_screen(joints, image_size=((1024, 1024),))[..., :2]
        err = torch.nn.functional.smooth_l1_loss(proj[0], gt_joints_2d[i, op_indices], reduction="none").sum(-1)
        loss_2d += (err * joint_conf[i, op_indices]).mean()
    
    reg_spine = torch.sum(body_pose[:, :21]**2) * 200.0 
    reg_pose  = torch.sum(body_pose**2) * 5.0 # 稍微放开约束，让动作更到位
    total_loss = loss_2d * 40 + reg_spine + reg_pose
    total_loss.backward(); opt2.step()
    
    if step % 200 == 0:
        print(f"Step {step:4d} | Total Loss: {total_loss.item():.2f} | 2D Pixel Error: {loss_2d.item():.2f}")

# 7. 最终结果汇总打印
print("\n" + "="*30)
print("🎯 最终拟合报告")
print("="*30)
print(f"最终 2D Loss (像素误差): {loss_2d.item():.4f}")
print(f"最终 Total Loss: {total_loss.item():.4f}")
print(f"模型保存位置: {SAVE_PATH}")
print(f"可视化检查图: {VIS_PATH_FINAL}")
print("="*30)

with torch.no_grad():
    out = smpl(body_pose=body_pose, global_orient=global_orient, transl=transl)
    save_obj(SAVE_PATH, out.vertices[0], smpl.faces_tensor)
    plot_debug(out, "Final Result", VIS_PATH_FINAL)