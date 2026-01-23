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

def generate_mesh_from_dem(dem_file, output_dir=None, ply_pre=True, target_epsg=4326):
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
    left = gt[0]
    res_x = gt[1]
    rot_x = gt[2]
    top = gt[3]
    rot_y = gt[4]
    res_y = gt[5]
    ncols = ds.RasterXSize
    nrows = ds.RasterYSize
    print(f'DEM size: {nrows} x {ncols}')
    
    # Generate points
    points = []
    point_indices = {}
    for r in range(nrows):
        for c in range(ncols):
            x = left + (c + 0.5) * res_x + (r + 0.5) * rot_x
            y = top + (r + 0.5) * res_y + (c + 0.5) * rot_y
            idx = len(points)
            points.append([x, y])
            point_indices[(r, c)] = idx
    points = np.array(points, dtype=np.float64)
    
    # Generate elements
    elements = []
    for r in range(nrows - 1):
        for c in range(ncols - 1):
            p_tl = point_indices[(r, c)]
            p_tr = point_indices[(r, c + 1)]
            p_bl = point_indices[(r + 1, c)]
            p_br = point_indices[(r + 1, c + 1)]
            elements.append([p_tl, p_br, p_tr])
            elements.append([p_tl, p_bl, p_br])
    elements = np.array(elements, dtype=np.int64)

    # Generate boundary dict
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

    # Generate elevation at points
    elev_points = elevation.flatten()
    nan_mask = np.isnan(elev_points)
    if np.any(nan_mask):
        elev_points[nan_mask] = np.nanmean(elev_points)
    
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
    
    ds = None

    return points, points_proj, elements, boundary, elev_points


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
