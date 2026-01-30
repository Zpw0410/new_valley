#!/usr/bin/env python3
import os
import numpy as np
from netCDF4 import Dataset
from osgeo import gdal
from tqdm import tqdm

# ================= 参数 =================
SWW_FILE = "DEM_Basin.sww"
OUT_DIR = "./depth2"
BASE_RES = 45.0  # 目标分辨率（m）
NODATA = -9999.0
EPSILON = 1e-3 * BASE_RES  # 容差，用于检测直角
# =======================================

def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    
    # ---------- 1. 读取 SWW ----------
    ds = Dataset(SWW_FILE, "r")
    x = ds.variables["x"][:]  # (Nodes,)
    y = ds.variables["y"][:]
    stage = ds.variables["stage"]  # (Time, Nodes)
    elev = ds.variables["elevation"][:]
    volumes = ds.variables["volumes"][:]  # 每个三角形的三个顶点索引
    depth = stage - elev  # 惰性计算
    ntime, nnode = stage.shape
    print(f"SWW nodes: {nnode}")
    print(f"Frames: {ntime}")
    
    # ---------- 2. 计算输出栅格范围 ----------
    xmin, xmax = x.min(), x.max()
    ymin, ymax = y.min(), y.max()
    cols = int(np.ceil((xmax - xmin) / BASE_RES))
    rows = int(np.ceil((ymax - ymin) / BASE_RES))
    print(f"Output raster: {rows} x {cols} @ {BASE_RES}m")
    
    # ---------- 3. GeoTransform ----------
    geotransform = (
        xmin, BASE_RES, 0.0,
        ymax, 0.0, -BASE_RES
    )
    driver = gdal.GetDriverByName("GTiff")
    
    # ---------- 4. 遍历每个时间步 ----------
    for t in tqdm(range(ntime), desc="Exporting depth"):
        grid = np.full((rows, cols), NODATA, dtype=np.float32)
        d = depth[t, :]
        d = np.maximum(d, 0.0)
        
        # 遍历每个三角形
        for tri_idx in range(volumes.shape[0]):
            tri = volumes[tri_idx]
            px = x[tri]  # [x1, x2, x3]
            py = y[tri]  # [y1, y2, y3]
            pd = d[tri]  # [d1, d2, d3]
            
            # 只处理等腰直角三角形（可选：如果你想处理所有三角形，可注释掉检测）
            right_vertex = -1
            for v in range(3):
                v1 = (v + 1) % 3
                v2 = (v + 2) % 3
                vec1 = (px[v1] - px[v], py[v1] - py[v])
                vec2 = (px[v2] - px[v], py[v2] - py[v])
                len1 = np.sqrt(vec1[0]**2 + vec1[1]**2)
                len2 = np.sqrt(vec2[0]**2 + vec2[1]**2)
                dot = vec1[0] * vec2[0] + vec1[1] * vec2[1]
                if abs(dot) < EPSILON and abs(len1 - len2) < EPSILON:
                    right_vertex = v
                    length = len1
                    span = round(length / BASE_RES)
                    if span not in [1, 2, 4]:
                        right_vertex = -1
                    break
            
            if right_vertex == -1:
                continue  # 跳过非等腰直角三角形
            
            # 计算栅格索引
            cols_idx = [int((px[i] - xmin) / BASE_RES) for i in range(3)]
            rows_idx = [int((ymax - py[i]) / BASE_RES) for i in range(3)]
            
            # 先填顶点（确保顶点精确）
            for v in range(3):
                r = rows_idx[v]
                c = cols_idx[v]
                if 0 <= r < rows and 0 <= c < cols:
                    grid[r, c] = max(grid[r, c], pd[v]) if grid[r, c] != NODATA else pd[v]
            
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
        ds_out.SetGeoTransform(geotransform)
        band = ds_out.GetRasterBand(1)
        band.WriteArray(grid)
        band.SetNoDataValue(NODATA)
        band.ComputeStatistics(False)
        ds_out.FlushCache()
        ds_out = None
    
    ds.close()
    print("✅ Finished exporting depth GeoTIFFs")

if __name__ == "__main__":
    main()