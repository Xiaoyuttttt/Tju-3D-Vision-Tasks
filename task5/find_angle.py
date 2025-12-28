import torch
import smplx
import os
import json
from pytorch3d.renderer import look_at_view_transform, FoVPerspectiveCameras

import numpy as np
# 终极补丁：覆盖所有 chumpy 可能调用的已弃用 numpy 别名
def patch_numpy():
    names = {
        'bool': bool,
        'int': int,
        'float': float,
        'complex': complex,
        'object': object,
        'str': str,
        'unicode': str,
        'nan': float('nan'),
        'inf': float('inf')
    }
    for name, obj in names.items():
        if not hasattr(np, name):
            setattr(np, name, obj)

patch_numpy()

device = torch.device("cuda:0")
MODEL_DIR = "/root/smpl_models_extracted/SMPL_python_v.1.1.0/smpl/models"
JSON_DIR = "/root/autodl-tmp/openpose_results"

# 1. 加载模型和数据
smpl = smplx.create(MODEL_DIR, model_type='smpl', gender='male', batch_size=1).to(device)
gt_joints_2d = []
joint_conf = []
for i in range(8):
    json_path = os.path.join(JSON_DIR, f"view_{i}_img_keypoints.json")
    with open(json_path, 'r') as f:
        data = json.load(f)
        points = np.array(data['people'][0]['pose_keypoints_2d']).reshape(-1, 3) if data['people'] else np.zeros((25, 3))
        gt_joints_2d.append(torch.from_numpy(points[:, :2]).float())
        joint_conf.append(torch.from_numpy(points[:, 2]).float())
gt_joints_2d = torch.stack(gt_joints_2d).to(device)
joint_conf = torch.stack(joint_conf).to(device)

# 2. 定义待测试的旋转方案 (X, Y, Z)
pi = 3.1415926
test_angles = {
    "原始 (0,0,0)": [0, 0, 0],
    "翻转180度 (X轴)": [pi, 0, 0],
    "翻转180度 (Z轴)": [0, 0, pi],
    "倒立且面朝后 (X,Y轴)": [pi, pi, 0],
    "侧转90度 (Z轴)": [0, 0, pi/2]
}

print(f"{'方案名称':<20} | {'初始 Loss':<15}")
print("-" * 40)

best_angle_name = ""
min_loss = float('inf')

for name, angle in test_angles.items():
    orient = torch.tensor([angle], dtype=torch.float32, device=device)
    # 固定 translator 在 (0, 2.5, 0)
    transl = torch.tensor([[0, 2.5, 0]], dtype=torch.float32, device=device)
    
    with torch.no_grad():
        output = smpl(global_orient=orient, transl=transl)
        joints_3d = output.joints[:, :25, :]
        
        current_loss = 0
        for i in range(8):
            R, T = look_at_view_transform(dist=11.0, elev=20, azim=i*45, at=((0, 2.5, 0),))
            cameras = FoVPerspectiveCameras(device=device, R=R, T=T)
            joints_2d_proj = cameras.transform_points_screen(joints_3d, image_size=((1024, 1024),))[..., :2]
            dist = torch.sum((joints_2d_proj[0] - gt_joints_2d[i])**2, dim=-1)
            current_loss += torch.mean(dist * joint_conf[i]).item()
            
    print(f"{name:<20} | {current_loss:<15.2f}")
    if current_loss < min_loss:
        min_loss = current_loss
        best_angle_name = name

print("-" * 40)
print(f"推荐方案: {best_angle_name} (Loss最低)")