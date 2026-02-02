#!/usr/bin/env python3
import os
import numpy as np
from netCDF4 import Dataset
from osgeo import gdal, osr
from tqdm import tqdm
from multiprocessing import Pool, cpu_count

# ================= 参数 =================
SWW_FILE = "DEM_Basin.sww"
OUT_DIR = "./depth2"
BASE_RES = 45.0  # 目标分辨率（m）
NODATA = -9999.0
# =======================================

# ---------------- Worker 全局上下文 ----------------
# 用于在 Windows / spawn 模式下存储每个进程独立的静态数据副本
worker_ctx = {}

def init_worker(sww_path, config):
    """
    子进程初始化函数
    在 spawn 模式下，每个进程都会执行此函数来加载必要的静态数据。
    虽然这会产生多份静态数据的内存副本，但相比于共享内存的复杂性，
    对于一般规模的网格（MB级），这是最稳健且跨平台的方案。
    """
    global worker_ctx
    
    # 1. 打开文件 (每个进程持有独立句柄)
    ds = Dataset(sww_path, "r")
    
    # 2. 读取静态数据到本地上下文
    worker_ctx['ds'] = ds
    worker_ctx['x'] = ds.variables["x"][:]
    worker_ctx['y'] = ds.variables["y"][:]
    worker_ctx['elev'] = ds.variables["elevation"][:]
    worker_ctx['volumes'] = ds.variables["volumes"][:]
    worker_ctx['config'] = config
    
def process_frame(t):
    """
    处理单帧：从全局上下文中读取数据，避免传递大对象
    """
    global worker_ctx
    
    # 从上下文中解包数据
    ds = worker_ctx['ds']
    x = worker_ctx['x']
    y = worker_ctx['y']
    elev = worker_ctx['elev']
    volumes = worker_ctx['volumes']
    cfg = worker_ctx['config']
    
    # ---------------- 核心计算 ----------------
    
    # 1. 按需读取当前时刻 Stage (切片读取，高效)
    stage_t = ds.variables["stage"][t, :]
    
    # 2. 计算深度
    d = stage_t - elev
    d = np.maximum(d, 0.0)
    
    # 3. 准备栅格
    rows, cols = cfg['rows'], cfg['cols']
    xmin, ymax = cfg['xmin'], cfg['ymax']
    base_res = cfg['base_res']
    nodata = cfg['nodata']
    
    grid = np.full((rows, cols), nodata, dtype=np.float32)
    
    # 4. 插值循环 (与原逻辑完全一致)
    for tri_idx in range(volumes.shape[0]):
        tri = volumes[tri_idx]
        px = x[tri]
        py = y[tri]
        pd = d[tri]
        
        # 计算栅格索引
        cols_idx = [int((px[i] - xmin) / base_res + 0.5) for i in range(3)]
        rows_idx = [int((ymax - py[i]) / base_res + 0.5) for i in range(3)]
        
        # ---------- 重心插值填充整个三角形 ----------
        x_a, y_a, d_a = px[0], py[0], pd[0]
        x_b, y_b, d_b = px[1], py[1], pd[1]
        x_c, y_c, d_c = px[2], py[2], pd[2]
        
        den = (y_b - y_c) * (x_a - x_c) + (x_c - x_b) * (y_a - y_c)
        if abs(den) < 1e-6:
            continue
        
        col_min = min(cols_idx)
        col_max = max(cols_idx)
        row_min = min(rows_idx)
        row_max = max(rows_idx)
        
        for r in range(row_min, row_max + 1):
            for c in range(col_min, col_max + 1):
                px_center = xmin + (c + 0.5) * base_res
                py_center = ymax - (r + 0.5) * base_res
                
                lambda_a = ((y_b - y_c) * (px_center - x_c) + (x_c - x_b) * (py_center - y_c)) / den
                lambda_b = ((y_c - y_a) * (px_center - x_c) + (x_a - x_c) * (py_center - y_c)) / den
                lambda_c = 1 - lambda_a - lambda_b
                
                if lambda_a >= -1e-6 and lambda_b >= -1e-6 and lambda_c >= -1e-6:
                    interp_d = lambda_a * d_a + lambda_b * d_b + lambda_c * d_c
                    if grid[r, c] == nodata:
                        grid[r, c] = interp_d
                    else:
                        grid[r, c] = max(grid[r, c], interp_d)

    # 5. 写入 GeoTIFF
    out_path = os.path.join(cfg['out_dir'], f"depth_{t:04d}.tif")
    driver = gdal.GetDriverByName("GTiff")
    if driver is None: return
    
    ds_out = driver.Create(out_path, cols, rows, 1, gdal.GDT_Float32)
    srs = osr.SpatialReference()
    srs.ImportFromEPSG(cfg['target_epsg'])
    ds_out.SetProjection(srs.ExportToWkt())
    ds_out.SetGeoTransform(cfg['geotransform'])
    
    band = ds_out.GetRasterBand(1)
    band.WriteArray(grid)
    band.SetNoDataValue(nodata)
    
    band.ComputeStatistics(False)
    ds_out.FlushCache()
    ds_out = None
    
    return t

def main(target_epsg: int):
    os.makedirs(OUT_DIR, exist_ok=True)
    
    # ---------- 1. 主进程读取元数据 (计算范围用) ----------
    print("Reading metadata...")
    ds = Dataset(SWW_FILE, "r")
    x = ds.variables["x"][:] 
    y = ds.variables["y"][:]
    ntime = ds.variables["stage"].shape[0]
    ds.close() # 立即关闭，释放主进程内存
    
    # ---------- 2. 计算输出范围 ----------
    delta = BASE_RES / 2.0
    xmin, xmax = x.min() - delta, x.max() + delta
    ymin, ymax = y.min() - delta, y.max() + delta
    cols = int(np.ceil((xmax - xmin) / BASE_RES) + 1.0)
    rows = int(np.ceil((ymax - ymin) / BASE_RES) + 1.0)
    print(f"Output raster: {rows} x {cols} @ {BASE_RES}m")
    
    # 清理掉 x, y 避免占用主进程内存
    del x, y
    
    # ---------- 3. 准备 Worker 配置 ----------
    geotransform = (xmin, BASE_RES, 0.0, ymax, 0.0, -BASE_RES)
    
    worker_config = {
        'rows': rows, 'cols': cols,
        'xmin': xmin, 'ymax': ymax,
        'base_res': BASE_RES, 'nodata': NODATA,
        'geotransform': geotransform,
        'target_epsg': target_epsg,
        'out_dir': OUT_DIR
    }
    
    # ---------- 4. 启动并行 ----------
    # 让系统自动决定，或手动指定
    num_workers = cpu_count()
    print(f"Starting parallel export with {num_workers} processes (Spawn-safe mode)...")
    
    # 关键：initializer 负责在每个子进程里各自加载静态数据
    with Pool(processes=num_workers, initializer=init_worker, initargs=(SWW_FILE, worker_config)) as pool:
        # 使用 imap_unordered 提高吞吐
        list(tqdm(pool.imap_unordered(process_frame, range(ntime)), total=ntime, desc="Exporting depth"))

    print("✅ Finished exporting depth GeoTIFFs")

if __name__ == '__main__':
    main(32615)