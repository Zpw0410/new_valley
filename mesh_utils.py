#!/usr/bin/env python3
# mesh_utils.py
"""
生成 ANUGA 网格（米单位）并输出对应 WGS84 经纬度，用于匹配降雨 TIF。
每个 DEM 像元正方形切成两个三角形（沿对角线）。
依赖：GDAL, numpy。
"""
import os
import numpy as np
from osgeo import gdal

def generate_mesh_from_dem(dem_file, output_dir=None, ply_pre=True):
    """
    从 DEM 文件生成 ANUGA 网格（米单位）并返回 WGS84 经纬度坐标。

    Returns
    -------
    points : np.ndarray (N,2)
        ANUGA 网格顶点（米单位）。
    points_lonlat : np.ndarray (N,2)
        对应 WGS84 经纬度坐标（度）。
    elements : np.ndarray (M,3)
        三角形索引。
    boundary : dict
        ANUGA 边界字典。
    elev_points : np.ndarray (N,)
        顶点高程。
    dx : float
        像元 x 方向等效米大小。
    dy : float
        像元 y 方向等效米大小。
    """
    # -----------------------------
    # 1️⃣ 打开 DEM
    # -----------------------------
    ds = gdal.Open(dem_file)
    if ds is None:
        raise RuntimeError(f"Cannot open DEM: {dem_file}")

    band = ds.GetRasterBand(1)
    elevation = band.ReadAsArray().astype(np.float32)
    nodata = band.GetNoDataValue()
    if nodata is not None:
        elevation = np.where(elevation == nodata, np.nan, elevation)

    gt = ds.GetGeoTransform()
    nrows, ncols = elevation.shape

    dx_deg = gt[1]
    dy_deg = abs(gt[5])
    xllcorner_deg = gt[0]
    yllcorner_deg = gt[3]

    # -----------------------------
    # 2️⃣ 度→米转换
    # -----------------------------
    lat_center = yllcorner_deg - (nrows * dy_deg) / 2.0
    deg2m_x = 111320.0 * np.cos(np.deg2rad(lat_center))
    deg2m_y = 110540.0

    dx = dx_deg * deg2m_x
    dy = dy_deg * deg2m_y

    # -----------------------------
    # 3️⃣ 构建顶点（米坐标）和索引字典
    # -----------------------------
    points = []
    points_lonlat = []
    point_index = {}
    for r in range(nrows):
        for c in range(ncols):
            # 像素中心米坐标
            x = c * dx
            y = -r * dy
            idx = len(points)
            points.append([x, y])
            point_index[(r, c)] = idx
            # 经纬度坐标（用于匹配降雨 TIF）
            lon = xllcorner_deg + c * dx_deg + dx_deg/2.0
            lat = yllcorner_deg - r * dy_deg - dy_deg/2.0
            points_lonlat.append([lon, lat])

    points = np.array(points)
    points_lonlat = np.array(points_lonlat)

    # -----------------------------
    # 4️⃣ 生成三角形，每个像元沿对角线切成两个三角形
    # -----------------------------
    elements = []
    for r in range(nrows - 1):
        for c in range(ncols - 1):
            p00 = point_index[(r, c)]
            p10 = point_index[(r, c + 1)]
            p01 = point_index[(r + 1, c)]
            p11 = point_index[(r + 1, c + 1)]
            # 对角线切分
            elements.append([p00, p10, p11])
            elements.append([p00, p11, p01])
    elements = np.array(elements, dtype=int)

    # -----------------------------
    # 5️⃣ 构建 ANUGA 边界
    # -----------------------------
    boundary = {}
    tri_id = 0
    for r in range(nrows - 1):
        for c in range(ncols - 1):
            # tri0
            if r == 0:
                boundary[(tri_id, 0)] = 'top'
            if c == ncols - 2:
                boundary[(tri_id, 1)] = 'right'
            tri_id += 1
            # tri1
            if c == 0:
                boundary[(tri_id, 0)] = 'left'
            if r == nrows - 2:
                boundary[(tri_id, 2)] = 'bottom'
            tri_id += 1

    # -----------------------------
    # 6️⃣ 高程映射
    # -----------------------------
    elev_points = elevation.flatten().astype(float)
    nan_mask = np.isnan(elev_points)
    if np.any(nan_mask):
        elev_points[nan_mask] = np.nanmean(elev_points)

    # -----------------------------
    # 7️⃣ 可选输出 PLY
    # -----------------------------
    if output_dir is not None and ply_pre:
        os.makedirs(output_dir, exist_ok=True)
        ply_file = os.path.join(output_dir, 'DEM_Basin_pre_domain.ply')
        with open(ply_file, 'w') as f:
            f.write("ply\nformat ascii 1.0\n")
            f.write(f"element vertex {points.shape[0]}\n")
            f.write("property float x\nproperty float y\nproperty float z\n")
            f.write(f"element face {elements.shape[0]}\n")
            f.write("property list uchar int vertex_indices\n")
            f.write("end_header\n")
            for i in range(points.shape[0]):
                f.write(f"{points[i,0]:.3f} {points[i,1]:.3f} {elev_points[i]:.3f}\n")
            for tri in elements:
                f.write(f"3 {tri[0]} {tri[1]} {tri[2]}\n")
        print("✔ 已生成 PLY 文件:", ply_file)

    return points, points_lonlat, elements, boundary, elev_points, dx, dy
