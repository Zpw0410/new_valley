#!/usr/bin/env python3
import os
import numpy as np
from netCDF4 import Dataset
from osgeo import gdal
import osgeo
from tqdm import tqdm
from multiprocessing import Pool, cpu_count
from functools import partial

# ================= 参数 =================
SWW_FILE = "DEM_Basin.sww"
OUT_DIR = "./depth2"
BASE_RES = 45.0  # 目标分辨率（m）
NODATA = -9999.0
# =======================================

def main(target_epsg: int):
    os.makedirs(OUT_DIR, exist_ok=True)
    
    # ---------- 1. 读取 SWW ----------
    ds = Dataset(SWW_FILE, "r")
    x = ds.variables["x"][:]  # (Nodes,)
    y = ds.variables["y"][:]
    stage = ds.variables["stage"]  # (Time, Nodes)
    elev = ds.variables["elevation"][:]
    volumes = ds.variables["volumes"][:]  # 每个三角形的三个顶点索引
    ntime, nnode = stage.shape
    print(f"SWW nodes: {nnode}")
    print(f"Frames: {ntime}")
    
    # ---------- 2. 计算输出栅格范围 ----------
    delta = BASE_RES / 2.0
    xmin, xmax = x.min() - delta, x.max() + delta
    ymin, ymax = y.min() - delta, y.max() + delta
    cols = int(np.ceil((xmax - xmin) / BASE_RES) + 1.0)
    rows = int(np.ceil((ymax - ymin) / BASE_RES) + 1.0)
    print(f"Output raster: {rows} x {cols} @ {BASE_RES}m")
    
    # ---------- 3. GeoTransform ----------
    geotransform = (
        xmin, BASE_RES, 0.0,
        ymax, 0.0, -BASE_RES
    )
    driver = gdal.GetDriverByName("GTiff")
    
    # ---------- 4. 遍历每个时间步 ----------
    for t in tqdm(range(ntime), desc="Exporting depth"):
        # Only output the last frame for testing
        if t != ntime - 1:
            continue
        
        grid = np.full((rows, cols), NODATA, dtype=np.float32)
        d = stage[t, :] - elev[:]
        d = np.maximum(d, 0.0)
        
        # 遍历每个三角形
        for tri_idx in range(volumes.shape[0]):
            tri = volumes[tri_idx]
            px = x[tri]  # [x1, x2, x3]
            py = y[tri]  # [y1, y2, y3]
            pd = d[tri]  # [d1, d2, d3]
            
            # 计算栅格索引
            cols_idx = [int((px[i] - xmin) / BASE_RES + 0.5) for i in range(3)]
            rows_idx = [int((ymax - py[i]) / BASE_RES + 0.5) for i in range(3)]
            
            # ---------- 重心插值填充整个三角形 ----------
            # A, B, C 对应三个顶点（顺序不重要）
            x_a, y_a, d_a = px[0], py[0], pd[0]
            x_b, y_b, d_b = px[1], py[1], pd[1]
            x_c, y_c, d_c = px[2], py[2], pd[2]
            
            # 计算分母（用于 Barycentric 坐标）
            den = (y_b - y_c) * (x_a - x_c) + (x_c - x_b) * (y_a - y_c)
            if abs(den) < 1e-6:
                continue  # 退化三角形
            
            # bounding box
            col_min = min(cols_idx)
            col_max = max(cols_idx)
            row_min = min(rows_idx)
            row_max = max(rows_idx)
            
            # 遍历 bounding box 内所有栅格
            for r in range(row_min, row_max + 1):
                for c in range(col_min, col_max + 1):
                    # 栅格中心坐标
                    px_center = xmin + (c + 0.5) * BASE_RES
                    py_center = ymax - (r + 0.5) * BASE_RES
                    
                    # 计算 Barycentric 坐标
                    lambda_a = ((y_b - y_c) * (px_center - x_c) + (x_c - x_b) * (py_center - y_c)) / den
                    lambda_b = ((y_c - y_a) * (px_center - x_c) + (x_a - x_c) * (py_center - y_c)) / den
                    lambda_c = 1 - lambda_a - lambda_b
                    
                    # 如果点在三角形内（包括边界）
                    if lambda_a >= -1e-6 and lambda_b >= -1e-6 and lambda_c >= -1e-6:
                        interp_d = lambda_a * d_a + lambda_b * d_b + lambda_c * d_c
                        if grid[r, c] == NODATA:
                            grid[r, c] = interp_d
                        else:
                            grid[r, c] = max(grid[r, c], interp_d)  # 或用平均，视需求
            
        # ---------- 5. 写 GeoTIFF ----------
        out_path = os.path.join(OUT_DIR, f"depth_{t:04d}.tif")
        ds_out = driver.Create(out_path, cols, rows, 1, gdal.GDT_Float32)
        
        # Set EPSG to target EPSG code
        srs = osgeo.osr.SpatialReference()
        srs.ImportFromEPSG(target_epsg)
        ds_out.SetProjection(srs.ExportToWkt())
        ds_out.SetGeoTransform(geotransform)
        
        band = ds_out.GetRasterBand(1)
        band.WriteArray(grid)
        band.SetNoDataValue(NODATA)
        band.ComputeStatistics(False)
        ds_out.FlushCache()
        ds_out = None
    
    ds.close()
    print("✅ Finished exporting depth GeoTIFFs")

if __name__ == '__main__':
    main(32615)