#!/usr/bin/env python3
# mesh_utils.py
"""
生成 ANUGA 网格（米单位）并输出对应 WGS84 经纬度，用于匹配降雨 TIF。
依赖：GDAL, anuga, numpy。
"""

import os
import numpy as np
from osgeo import gdal
import anuga
from anuga import rectangular_cross


def generate_mesh_from_dem(dem_file, output_dir=None, ply_pre=True):
    """
    从 DEM 文件生成 ANUGA 网格（米单位）并返回 WGS84 经纬度坐标。

    Parameters
    ----------
    dem_file : str
        DEM 文件路径（WGS84 tif）。
    output_dir : str or None
        若提供，则输出 pre-domain PLY。
    ply_pre : bool
        是否输出 pre-domain PLY（默认 True）。

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
    # 2️⃣ 度→米转换（ANUGA 网格用）
    # -----------------------------
    lat_center = yllcorner_deg - (nrows * dy_deg) / 2.0
    deg2m_x = 111320.0 * np.cos(np.deg2rad(lat_center))
    deg2m_y = 110540.0
    dx = dx_deg * deg2m_x
    dy = dy_deg * deg2m_y

    # -----------------------------
    # 3️⃣ 生成 ANUGA 网格
    # -----------------------------
    m = ncols - 1
    n = nrows - 1
    len1 = m * dx
    len2 = n * dy
    x_min, y_max = 0.0, 0.0
    y_min = y_max - nrows * dy
    origin = (x_min, y_min)

    points, elements, boundary = rectangular_cross(
        m, n, len1=len1, len2=len2, origin=origin
    )

    # -----------------------------
    # 4️⃣ raster 高程映射到 ANUGA 点
    # -----------------------------
    cols = np.round((points[:, 0] - x_min) / dx).astype(int)
    rows = np.round((y_max - points[:, 1]) / dy).astype(int)
    cols = np.clip(cols, 0, ncols - 1)
    rows = np.clip(rows, 0, nrows - 1)

    elev_points = elevation[rows, cols].astype(float)
    nan_mask = np.isnan(elev_points)
    if np.any(nan_mask):
        elev_points[nan_mask] = np.nanmean(elev_points)

    # -----------------------------
    # 5️⃣ 生成 WGS84 经纬度
    # -----------------------------
    lon_points = xllcorner_deg + cols * dx_deg + dx_deg / 2.0
    lat_points = yllcorner_deg - rows * dy_deg - dy_deg / 2.0
    points_lonlat = np.column_stack([lon_points, lat_points])

    # -----------------------------
    # 6️⃣ 可选输出 pre-domain PLY
    # -----------------------------
    if output_dir is not None and ply_pre:
        os.makedirs(output_dir, exist_ok=True)
        pre_ply = os.path.join(output_dir, 'DEM_Basin_pre_domain.ply')

        with open(pre_ply, 'w') as f:
            f.write("ply\n")
            f.write("format ascii 1.0\n")
            f.write(f"element vertex {points.shape[0]}\n")
            f.write("property float x\n")
            f.write("property float y\n")
            f.write("property float z\n")
            f.write(f"element face {elements.shape[0]}\n")
            f.write("property list uchar int vertex_indices\n")
            f.write("end_header\n")

            # 写入顶点
            for i in range(points.shape[0]):
                f.write(f"{points[i,0]:.3f} {points[i,1]:.3f} {elev_points[i]:.3f}\n")

            # 写入三角形
            for tri in elements:
                f.write(f"3 {tri[0]} {tri[1]} {tri[2]}\n")

        print("✔ 已生成 PLY 文件:", pre_ply)

    return points, points_lonlat, elements, boundary, elev_points, dx, dy
