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
# import pyproj
# pyproj.datadir.set_data_dir(r"D:\Anaconda\envs\anuga_env\Library\share\proj")

def _pixel_center_from_gt(gt, r, c):
    """计算像素中心坐标，支持旋转 geotransform"""
    x = gt[0] + (c + 0.5) * gt[1] + (r + 0.5) * gt[2]
    y = gt[3] + (c + 0.5) * gt[4] + (r + 0.5) * gt[5]
    return x, y

def generate_mesh_from_dem(dem_file, output_dir=None, ply_pre=True, target_epsg=4326):
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    ds = gdal.Open(dem_file)
    if ds is None:
        raise RuntimeError(f"Cannot open DEM: {dem_file}")

    gt = ds.GetGeoTransform()
    left = gt[0]
    res_x = gt[1]
    rot_x = gt[2]
    top = gt[3]
    rot_y = gt[4]
    res_y = gt[5]
    ncols = ds.RasterXSize
    nrows = ds.RasterYSize
    print(f'DEM size: {nrows} x {ncols}')

    band = ds.GetRasterBand(1)
    elevation = band.ReadAsArray().astype(np.float64)
    nodata = band.GetNoDataValue()
    if nodata is not None:
        elevation = np.where(elevation == nodata, np.nan, elevation)
    
    # Generate points
    points = []
    filtered_elevation = []
    point_indices = {}
    for r in range(nrows):
        for c in range(ncols):
            x = left + (c + 0.5) * res_x + (r + 0.5) * rot_x
            y = top + (r + 0.5) * res_y + (c + 0.5) * rot_y
            z = elevation[r, c]
            if np.isnan(z):
                continue
            idx = len(points)
            points.append([x, y])
            filtered_elevation.append(z)
            point_indices[(r, c)] = idx
    points = np.array(points, dtype=np.float64)
    filtered_elevation = np.array(filtered_elevation, dtype=np.float64).flatten()
    
    # Generate elements
    elements = []
    
    # 用于追踪边的使用次数，以此构建拓扑边界
    # Key: 边的顶点对 (sorted tuple), Value: list of (tri_id, edge_id, direction_tag)
    # direction_tag 仅用于当该边最终判定为边界时，赋予其物理意义上的方位（top/bottom/left/right）
    edge_tracker = {}
    
    def add_edge_usage(v1, v2, tri_id, edge_id, tag):
        # 对顶点排序，确保 (u, v) 和 (v, u) 被视为同一条边
        edge_key = tuple(sorted((v1, v2)))
        if edge_key not in edge_tracker:
            edge_tracker[edge_key] = []
        edge_tracker[edge_key].append((tri_id, edge_id, tag))
    
    for r in range(nrows - 1):
        for c in range(ncols - 1):
            p_tl = point_indices.get((r, c), None)
            p_tr = point_indices.get((r, c + 1), None)
            p_bl = point_indices.get((r + 1, c), None)
            p_br = point_indices.get((r + 1, c + 1), None)
            
            # is_tl_nan = np.isnan(elevation[r, c])
            # is_tr_nan = np.isnan(elevation[r, c + 1])
            # is_bl_nan = np.isnan(elevation[r + 1, c])
            # is_br_nan = np.isnan(elevation[r + 1, c + 1])
            
            # if not (is_tl_nan or is_tr_nan or is_br_nan):
            if not (p_tl is None or p_tr is None or p_br is None):
                elements.append([p_tl, p_br, p_tr])
                tid = len(elements) - 1
                
                # Edge0: BR(1)-TR(2)
                add_edge_usage(p_br, p_tr, tid, 0, 'right')
                
                # Edge1: TR(2)-TL[0]
                add_edge_usage(p_tr, p_tl, tid, 1, 'top')
                
                # Edge2: TL(0)-BR(1)
                add_edge_usage(p_tl, p_br, tid, 2, 'exterior')
                
            if not (p_tl is None or p_bl is None or p_br is None):
                elements.append([p_tl, p_bl, p_br])
                tid = len(elements) - 1
                
                # Edge0: BL(1)-BR(2)
                add_edge_usage(p_bl, p_br, tid, 0, 'bottom')
                
                # Edge1: BR(2)-TL(0)
                add_edge_usage(p_br, p_tl, tid, 1, 'exterior')
                
                # Edge2: TL(0)-BL(1)
                add_edge_usage(p_tl, p_bl, tid, 2, 'left')
                
    elements = np.array(elements, dtype=np.int64)
    
    # Generate boundary dict based on topology
    # 核心逻辑：只有当 usages 列表长度为 1 时，说明该边没有邻居，是边界边
    boundary = {}
    for edge_key, usages in edge_tracker.items():
        if len(usages) == 1:
            tri_id, edge_id, tag = usages[0]
            boundary[(tri_id, edge_id)] = tag

    # # Generate elevation at points
    # elev_points = elevation.flatten()
    # nan_mask = np.isnan(elev_points)
    # if np.any(nan_mask):
    #     elev_points[nan_mask] = np.nanmean(elev_points)
    
    # Get epsg code from DEM
    proj_wkt = ds.GetProjectionRef()
    source_crs = gdal.osr.SpatialReference()
    source_crs.ImportFromWkt(proj_wkt)
    source_epsg = int(source_crs.GetAttrValue('AUTHORITY', 1))
    
    # Reproject points to target_epsg
    transformer = Transformer.from_crs(f"EPSG:{source_epsg}", f"EPSG:{target_epsg}", always_xy=True)
    points_proj = np.array(transformer.transform(points[:,0], points[:,1])).T

    # Yield PLY
    if output_dir and ply_pre:
        ply_file = os.path.join(output_dir, 'DEM_Basin_pre_domain.ply')
        with open(ply_file, 'w') as f:
            f.write("ply\nformat ascii 1.0\n")
            f.write(f"element vertex {points.shape[0]}\n")
            f.write("property float x\nproperty float y\nproperty float z\n")
            f.write(f"element face {elements.shape[0]}\n")
            f.write("property list uchar int vertex_indices\n")
            f.write("end_header\n")
            for i in range(points.shape[0]):
                f.write(f"{points[i,0]:.3f} {points[i,1]:.3f} {filtered_elevation[i]:.3f}\n")
            for tri in elements:
                f.write(f"3 {tri[0]} {tri[1]} {tri[2]}\n")
        print("✔ 已生成 PLY 文件:", ply_file)
    
    ds = None

    return points, points_proj, elements, boundary, filtered_elevation


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
