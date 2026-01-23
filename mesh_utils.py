import os
import numpy as np

def generate_rect_mesh(elevation, geotransform, output_dir=None, ply_name='mesh_pre_domain.ply'):
    """
    根据 DEM 高程生成规则三角形网格，并可导出 PLY 文件。
    每个正方形切成两个三角形。
    
    Parameters
    ----------
    elevation : np.ndarray
        DEM 高程数组 (nrows x ncols)
    geotransform : tuple
        GDAL 仿射变换参数 (origin_x, px_w, rot_x, origin_y, rot_y, px_h)
    output_dir : str, optional
        如果提供，则导出 PLY 文件
    ply_name : str
        PLY 文件名

    Returns
    -------
    points : np.ndarray
        顶点坐标数组 (N x 2)，单位与 DEM 一致（经纬度或投影）
    elements : np.ndarray
        三角形索引数组 (M x 3)
    boundary : dict
        ANUGA 边界字典
    elev_points : np.ndarray
        顶点高程
    """
    nrows, ncols = elevation.shape
    origin_x, px_w, rot_x, origin_y, rot_y, px_h = geotransform

    # -----------------------------
    # 1️⃣ 构建规则点阵 (使用像素中心坐标)
    # -----------------------------
    points = []
    point_index = {}
    for r in range(nrows):
        for c in range(ncols):
            # 像素中心坐标
            x = origin_x + (c + 0.5) * px_w + (r + 0.5) * rot_x
            y = origin_y + (c + 0.5) * rot_y + (r + 0.5) * px_h
            idx = len(points)
            points.append([x, y])
            point_index[(r, c)] = idx
    points = np.array(points)

    # -----------------------------
    # 2️⃣ 每个正方形切成 2 个三角形
    # -----------------------------
    elements = []
    for r in range(nrows - 1):
        for c in range(ncols - 1):
            p00 = point_index[(r, c)]
            p10 = point_index[(r, c + 1)]
            p01 = point_index[(r + 1, c)]
            p11 = point_index[(r + 1, c + 1)]

            # 对角线切分
            elements.append([p00, p10, p11])  # tri0
            elements.append([p00, p11, p01])  # tri1
    elements = np.array(elements, dtype=int)

    # -----------------------------
    # 3️⃣ 构建 boundary（严格按照 ANUGA edge_id 规则）
    # -----------------------------
    boundary = {}
    tri_id = 0
    for r in range(nrows - 1):
        for c in range(ncols - 1):
            if r == 0:
                boundary[(tri_id, 0)] = 'top'
            if c == ncols - 2:
                boundary[(tri_id, 1)] = 'right'
            tri_id += 1

            if c == 0:
                boundary[(tri_id, 0)] = 'left'
            if r == nrows - 2:
                boundary[(tri_id, 2)] = 'bottom'
            tri_id += 1

    # -----------------------------
    # 4️⃣ 高程映射到顶点
    # -----------------------------
    elev_points = elevation.flatten().astype(float)
    nan_mask = np.isnan(elev_points)
    if np.any(nan_mask):
        elev_points[nan_mask] = np.nanmean(elev_points)

    # -----------------------------
    # 5️⃣ 可选: 输出 PLY 文件
    # -----------------------------
    if output_dir is not None:
        os.makedirs(output_dir, exist_ok=True)
        ply_file = os.path.join(output_dir, ply_name)
        with open(ply_file, 'w') as f:
            f.write("ply\n")
            f.write("format ascii 1.0\n")
            f.write(f"element vertex {points.shape[0]}\n")
            f.write("property float x\nproperty float y\nproperty float z\n")
            f.write(f"element face {elements.shape[0]}\n")
            f.write("property list uchar int vertex_indices\n")
            f.write("end_header\n")
            for i in range(points.shape[0]):
                f.write(f"{points[i,0]:.6f} {points[i,1]:.6f} {elev_points[i]:.3f}\n")
            for tri in elements:
                f.write(f"3 {tri[0]} {tri[1]} {tri[2]}\n")
        print("✔ 已生成 PLY 文件:", ply_file)

    return points, elements, boundary, elev_points
