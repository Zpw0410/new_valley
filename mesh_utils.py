#!/usr/bin/env python3
# mesh_utils.py
"""
从 WGS84 DEM 生成 ANUGA 网格，并把顶点投影到 EPSG:32615（米单位）。

依赖：
- gdal
- numpy
- pyproj
"""
import os
import numpy as np
from osgeo import gdal
from pyproj import Transformer

def _pixel_center_from_gt(gt, r, c):
    """计算像素中心坐标，支持旋转 geotransform"""
    x = gt[0] + (c + 0.5) * gt[1] + (r + 0.5) * gt[2]
    y = gt[3] + (c + 0.5) * gt[4] + (r + 0.5) * gt[5]
    return x, y

def generate_mesh_from_dem(dem_file, output_dir=None, ply_pre=True, target_epsg=32615):
    """
    生成网格，并投影到 target_epsg。

    Returns
    -------
    points_proj : Nx2 array, EPSG:target_epsg 坐标（米单位）
    points_lonlat : Nx2 array, 原始经纬度
    elements : Mx3 array, 三角形索引
    boundary : dict, ANUGA 边界
    elev_points : N array, 顶点高程
    dx : float, 米方向步长（x）
    dy : float, 米方向步长（y）
    """
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    ds = gdal.Open(dem_file)
    if ds is None:
        raise RuntimeError(f"Cannot open DEM: {dem_file}")

    band = ds.GetRasterBand(1)
    elevation = band.ReadAsArray().astype(np.float64)
    nodata = band.GetNoDataValue()
    if nodata is not None:
        elevation = np.where(elevation == nodata, np.nan, elevation)

    gt = ds.GetGeoTransform()
    ncols = ds.RasterXSize
    nrows = ds.RasterYSize

    # 1️⃣ 生成经纬度顶点
    points_lonlat = []
    point_index = {}
    for r in range(nrows):
        for c in range(ncols):
            lon, lat = _pixel_center_from_gt(gt, r, c)
            idx = len(points_lonlat)
            points_lonlat.append([lon, lat])
            point_index[(r, c)] = idx
    points_lonlat = np.array(points_lonlat, dtype=np.float64)

    # 2️⃣ 生成三角形
    elements = []
    for r in range(nrows - 1):
        for c in range(ncols - 1):
            p00 = point_index[(r, c)]
            p10 = point_index[(r, c + 1)]
            p01 = point_index[(r + 1, c)]
            p11 = point_index[(r + 1, c + 1)]
            elements.append([p00, p10, p11])
            elements.append([p00, p11, p01])
    elements = np.array(elements, dtype=int)

    # 3️⃣ 边界字典
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

    # 4️⃣ 高程
    elev_points = elevation.flatten()
    nan_mask = np.isnan(elev_points)
    if np.any(nan_mask):
        elev_points[nan_mask] = np.nanmean(elev_points)

    # 5️⃣ 投影到 EPSG:32615
    transformer = Transformer.from_crs("EPSG:4326", f"EPSG:{target_epsg}", always_xy=True)
    points_proj = np.array(transformer.transform(points_lonlat[:,0], points_lonlat[:,1])).T

    # 6️⃣ 估算 dx, dy
    dx = dy = None
    if ncols > 1 and nrows > 1:
        dx = np.mean(np.linalg.norm(points_proj[1:ncols, :] - points_proj[0:ncols-1, :], axis=1))
        dy = np.mean(np.linalg.norm(points_proj[ncols:2*ncols, :] - points_proj[0:ncols, :], axis=1))

    # 7️⃣ 输出 PLY
    if output_dir and ply_pre:
        ply_file = os.path.join(output_dir, 'DEM_Basin_pre_domain.ply')
        with open(ply_file, 'w') as f:
            f.write("ply\nformat ascii 1.0\n")
            f.write(f"element vertex {points_proj.shape[0]}\n")
            f.write("property float x\nproperty float y\nproperty float z\n")
            f.write(f"element face {elements.shape[0]}\n")
            f.write("property list uchar int vertex_indices\n")
            f.write("end_header\n")
            for i in range(points_proj.shape[0]):
                f.write(f"{points_proj[i,0]:.3f} {points_proj[i,1]:.3f} {elev_points[i]:.3f}\n")
            for tri in elements:
                f.write(f"3 {tri[0]} {tri[1]} {tri[2]}\n")
        print("✔ 已生成 PLY 文件:", ply_file)

    return points_proj, points_lonlat, elements, boundary, elev_points, dx, dy


# Example usage
# if __name__ == "__main__":
#     dem_file = "path/to/your/wgs84_dem.tif"
#     output_dir = "./mesh_out"
#     pts_proj, pts_lonlat, elems, bnd, elevs, dx_m, dy_m = generate_mesh_from_dem(
#         dem_file, output_dir=output_dir, ply_pre=True, target_epsg=32615
#     )
#     print("Done.")
#     print("points_proj.shape:", pts_proj.shape)
#     print("dx ~", dx_m, "m, dy ~", dy_m, "m")
