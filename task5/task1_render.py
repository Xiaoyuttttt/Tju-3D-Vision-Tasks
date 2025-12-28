import os
import torch
import cv2
import numpy as np
from pytorch3d.io import load_objs_as_meshes
from pytorch3d.renderer import (
    look_at_view_transform,
    FoVPerspectiveCameras, 
    PointLights, 
    RasterizationSettings, 
    MeshRenderer, 
    MeshRasterizer,  
    SoftPhongShader
)

device = torch.device("cuda:0")
OBJ_PATH = "/root/autodl-tmp/126111535847796.obj"
SAVE_DIR = "/root/autodl-tmp/render_results"
os.makedirs(SAVE_DIR, exist_ok=True)

mesh = load_objs_as_meshes([OBJ_PATH], device=device)

# 渲染配置
raster_settings = RasterizationSettings(image_size=1024, blur_radius=0.0, faces_per_pixel=1)

print("正在重新渲染全身视图（尝试进一步抬高和拉远相机）...")
for i in range(8):
    azim = i * 45
    # 修改重点：dist=5.0 (拉远), at=((0, 1.5, 0)) (抬高到胸部高度)
    R, T = look_at_view_transform(dist=11.0, elev=20, azim=azim, at=((0, 2.5, 0),))
    cameras = FoVPerspectiveCameras(device=device, R=R, T=T)

    renderer = MeshRenderer(
        rasterizer=MeshRasterizer(cameras=cameras, raster_settings=raster_settings),
        shader=SoftPhongShader(device=device, cameras=cameras, lights=PointLights(device=device, location=[[0, 5, -5]]))
    )

    images = renderer(mesh) 
    rgba = images[0].cpu().numpy()
    img = (rgba[..., :3] * 255).astype(np.uint8)
    mask = (rgba[..., 3] > 0).astype(np.uint8) * 255

    cv2.imwrite(os.path.join(SAVE_DIR, f"view_{i}_img.jpg"), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
    cv2.imwrite(os.path.join(SAVE_DIR, f"view_{i}_mask.png"), mask)
    print(f"✅ 视角 {i} 已保存")

print("渲染修正完成！请刷新 Jupyter 左侧文件列表查看 view_0_img.jpg")
