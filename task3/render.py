import numpy as np
import cv2
import os

def main():
    # 创建一张白色画布（800x800）
     # 1. 创建画布
    H, W = 800, 800
    canvas = np.ones((H, W, 3), dtype=np.uint8) * 255

    # 2. 读取 OBJ 顶点
    obj_path = "dolphin_without_lefthand.obj"  # 确保路径正确
    vertices = load_obj_vertices(obj_path)

    # 2. 归一化
    vertices = normalize_vertices(vertices)
    vertices = np.asarray(vertices, dtype=np.float32)


    # # 3. 验证归一化效果
    # print("归一化后 y 最小值:", vertices[:, 1].min())
    # print("归一化后 y 最大值:", vertices[:, 1].max())
    # print("归一化后重心:", np.mean(vertices, axis=0))

    os.makedirs("output", exist_ok=True)
    cv2.imwrite("output/step2_empty.png", canvas)
    
    mode = 2  # 1 or 2

    if mode == 1:
        vertices_cam = apply_camera_param1(vertices)
    else:
        vertices_cam = apply_camera_param2(
            vertices,
            theta_deg=45,
            phi_deg=45,
            r=2.25
        )
    
    # === 投影 ===
    f = 2.5
    points_2d = project_points(vertices_cam, f, (H, W))

    # === 画点 ===
    for u, v in points_2d:
        cv2.circle(canvas, (u, v), 1, (0, 0, 0), -1)

    faces = load_obj_faces(obj_path)

    # === 保存 ===
    if mode == 1:
        cv2.imwrite("output/mode1.png", canvas)
    else:
        draw_wireframe(
            canvas,
            vertices_cam,
            faces,
            f=2.5,
            img_size=(H, W)
        )
        cv2.imwrite("output/mode2.png", canvas)



def load_obj_vertices(obj_path):
    vertices = []

    with open(obj_path, 'r') as f:
        for line in f:
            if line.startswith('v '):
                parts = line.strip().split()
                x, y, z = map(float, parts[1:4])
                vertices.append([x, y, z])

    vertices = np.array(vertices, dtype=np.float32)
    return vertices


def normalize_vertices(vertices):
    """
    vertices: (N, 3)
    return:   归一化后的 vertices
    """

    # 1. 计算重心
    center = np.mean(vertices, axis=0)

    # 2. 平移到原点
    vertices_centered = vertices - center

    # 3. 计算 y 方向高度
    y_min = vertices_centered[:, 1].min()
    y_max = vertices_centered[:, 1].max()
    height = y_max - y_min

    # 4. 缩放，使 y 方向高度 = 1
    vertices_normalized = vertices_centered / height

    return vertices_normalized

def project_points(vertices, f, img_size):
    """
    vertices: (N, 3) 相机坐标系下
    f: 焦距
    img_size: (H, W)
    """
    H, W = img_size
    projected_points = []

    for x, y, z in vertices:
        # 只投影在相机前方的点
        if z <= 0:
            continue

        # 小孔成像
        x_img = f * x / z
        y_img = f * y / z

        # 只保留在 [-1, 1] 范围内的点
        if abs(x_img) > 1 or abs(y_img) > 1:
            continue

        # 图像坐标 → 像素坐标
        u = int((x_img + 1) * W / 2)
        v = int((1 - y_img) * H / 2)

        projected_points.append((u, v))

    return projected_points


def load_obj_faces(obj_path):
    faces = []

    with open(obj_path, 'r') as f:
        for line in f:
            if line.startswith('f '):
                parts = line.strip().split()
                # 只取顶点索引，OBJ 是从 1 开始的
                idx = [int(p.split('/')[0]) - 1 for p in parts[1:4]]
                faces.append(idx)

    return faces

def apply_camera_param1(vertices):
    """
    相机参数 1：
    R = I
    C = (0, 0, -2.25)
    """
    R = np.eye(3, dtype=np.float32)
    C = np.array([0,0,-2.25])
    vertices_cam = world_to_camera(vertices, R, C)

    return vertices_cam

def world_to_camera(vertices, R, C):
        """
        vertices: (N,3) 世界坐标
        R: (3,3) 旋转矩阵
        C: (3,)   相机中心（世界坐标）
        """
        return (R @ (vertices - C).T).T
    
def look_at(camera_pos, target, up=np.array([0,1,0], dtype=np.float32)):
        z = target - camera_pos
        z /= np.linalg.norm(z)

        x = np.cross(up, z)
        x /= np.linalg.norm(x)

        y = np.cross(z, x)

        R = np.stack([x, y, z], axis=0)
        return R

def apply_camera_param2(vertices, theta_deg, phi_deg, r):
    theta = np.deg2rad(theta_deg)
    phi = np.deg2rad(phi_deg)

    camera_pos = np.array([
        r * np.cos(phi) * np.sin(theta),
        r * np.sin(phi),
        r * np.cos(phi) * np.cos(theta)
    ], dtype=np.float32)

    target = np.array([0.0, 0.0, 0.0], dtype=np.float32)

    R = look_at(camera_pos, target)

    vertices_cam = world_to_camera(vertices, R, camera_pos)
    return vertices_cam


def draw_wireframe(canvas, vertices_cam, faces, f, img_size):
    H, W = img_size

    # 1. 投影所有顶点（保留 index）
    projected = {}

    for idx, (x, y, z) in enumerate(vertices_cam):
        if z <= 0:
            continue

        x_img = f * x / z
        y_img = f * y / z

        if abs(x_img) > 1 or abs(y_img) > 1:
            continue

        u = int((x_img + 1) * W / 2)
        v = int((1 - y_img) * H / 2)

        projected[idx] = (u, v)

    # 2. 从 faces 构造边
    edges = set()
    for i, j, k in faces:
        edges.add(tuple(sorted((i, j))))
        edges.add(tuple(sorted((j, k))))
        edges.add(tuple(sorted((k, i))))

    # 3. 画边
    for i, j in edges:
        if i in projected and j in projected:
            cv2.line(canvas, projected[i], projected[j], (0, 0, 0), 1)




if __name__ == "__main__":
    main()