import os
import numpy as np
import trimesh
import pyrender
import subprocess

# ---------- 配置 ----------
OBJ_PATH = "../task2/house.obj"
OUTPUT_DIR = "./output"
VIDEO_PATH = os.path.join(OUTPUT_DIR, "house_rotation.mp4")
RESOLUTION = (800, 800)
FPS = 12
FRAMES_COUNT = 36


FFMPEG_PATH = r"D:\ffmpeg\ffmpeg-master-latest-win64-gpl-shared\bin\ffmpeg.exe"
# ------------------------

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ---------- 清理 OBJ 文件 ----------
def clean_obj(obj_path, tmp_path="__clean.obj"):
    with open(obj_path, "r", encoding="utf-8", errors="ignore") as fin, \
         open(tmp_path, "w", encoding="utf-8") as fout:
        for line in fin:
            line = line.strip()
            if line.startswith(("v ", "vt ", "vn ", "f ", "mtllib ", "usemtl ")):
                if '#' in line:
                    line = line.split('#')[0].rstrip()
                fout.write(line + "\n")
    return tmp_path

OBJ_CLEAN = clean_obj(OBJ_PATH)

# ---------- 加载 OBJ ----------
mesh_or_scene = trimesh.load(OBJ_CLEAN, process=False)

if isinstance(mesh_or_scene, trimesh.Scene):
    combined = trimesh.util.concatenate([g for g in mesh_or_scene.geometry.values()])
    pyr_meshes = [pyrender.Mesh.from_trimesh(g, smooth=False)
                  for g in mesh_or_scene.geometry.values()]
else:
    combined = mesh_or_scene
    pyr_meshes = [pyrender.Mesh.from_trimesh(mesh_or_scene, smooth=False)]

# ---------- 创建场景 ----------
scene = pyrender.Scene()
for m in pyr_meshes:
    scene.add(m)

# ---------- 光源 ----------
light = pyrender.PointLight(color=[1,1,1], intensity=5.0)
scene.add(light, pose=np.eye(4))

# ---------- 相机 ----------
camera = pyrender.PerspectiveCamera(yfov=np.pi/3.0)
camera_node = scene.add(camera, pose=np.eye(4))  # 保存节点

radius = np.max(combined.extents) * 3

# ---------- Offscreen 渲染 ----------
r = pyrender.OffscreenRenderer(viewport_width=RESOLUTION[0],
                               viewport_height=RESOLUTION[1])

frames = []

for i, angle in enumerate(np.linspace(0, 2*np.pi, FRAMES_COUNT)):
    camera_pose = trimesh.transformations.compose_matrix(
        angles=[0, angle, 0],
        translate=[0, 0, radius]
    )
    scene.set_pose(camera_node, camera_pose)
    color, _ = r.render(scene)
    frames.append(color)
    print(f"Rendered frame {i+1}/{FRAMES_COUNT}")

r.delete()

# ---------- 保存视频，使用 ffmpeg ----------
def save_video_ffmpeg(frames, path, fps=12):
    h, w, _ = frames[0].shape
    cmd = [
        FFMPEG_PATH,
        '-y',
        '-f', 'rawvideo',
        '-vcodec', 'rawvideo',
        '-pix_fmt', 'rgb24',
        '-s', f'{w}x{h}',
        '-r', str(fps),
        '-i', '-',   # 从 stdin 输入
        '-an',
        '-vcodec', 'libx264',
        '-pix_fmt', 'yuv420p',
        path
    ]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    for frame in frames:
        proc.stdin.write(frame.astype(np.uint8).tobytes())
    proc.stdin.close()
    proc.wait()

save_video_ffmpeg(frames, VIDEO_PATH, fps=FPS)
print("✅ 视频保存完成：", VIDEO_PATH)
