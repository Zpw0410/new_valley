#!/usr/bin/env python3
import os
import numpy as np
from netCDF4 import Dataset
from osgeo import gdal, osr
from tqdm import tqdm

# ================= 参数 =================
SWW_FILE = "DEM_Basin.sww"
OUT_DIR = "./depth_geotiff"
BASE_RES = 45.0        # 目标分辨率（m）
NODATA = -9999.0
# =======================================

def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    # ---------- 1. 读取 SWW ----------
    ds = Dataset(SWW_FILE, "r")

    x = ds.variables["x"][:]        # (Nodes,)
    y = ds.variables["y"][:]
    stage = ds.variables["stage"]   # (Time, Nodes)
    elev = ds.variables["elevation"][:]

    depth = stage - elev            # 惰性计算
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

    # ---------- 4. 时间循环 ----------
    for t in tqdm(range(ntime), desc="Exporting depth"):
        grid = np.full((rows, cols), NODATA, dtype=np.float32)

        d = depth[t, :]
        d = np.maximum(d, 0.0)

        for i in range(nnode):
            xi, yi = x[i], y[i]

            # 对齐到 45m 网格索引
            col = int((xi - xmin) // BASE_RES)
            row = int((ymax - yi) // BASE_RES)

            # 估算该点所属“原始网格尺度”
            # 通过相邻点距离判断
            # 假设 ANUGA 节点顺序无乱序
            if i < nnode - 1:
                dx = abs(x[i+1] - xi)
                dy = abs(y[i+1] - yi)
                size = max(dx, dy)
            else:
                size = BASE_RES

            if size <= 50:
                span = 1        # 45m
            elif size <= 100:
                span = 2        # 90m
            else:
                span = 4        # 180m

            for rr in range(span):
                for cc in range(span):
                    r = row + rr
                    c = col + cc
                    if 0 <= r < rows and 0 <= c < cols:
                        grid[r, c] = d[i]

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
