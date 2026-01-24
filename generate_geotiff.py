#!/usr/bin/env python3
import os
import numpy as np
from netCDF4 import Dataset
from osgeo import gdal, osr
from tqdm import tqdm
from config import REPROJECTED_DEM_FILE

# ================= 配置 =================
SWW_FILE = r'./DEM_Basin.sww'
OUT_DIR = r'./geotiff_output'
QUANTITY = 'depth'  # 输出水深
# =======================================

def get_variable(ds, candidates):
    for name in candidates:
        if name in ds.variables:
            return ds.variables[name][:]
    return None

def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    
    # 1. 打开原始参考 DEM 获取地理信息
    if not os.path.exists(REPROJECTED_DEM_FILE):
        raise FileNotFoundError(f"Reference DEM not found: {REPROJECTED_DEM_FILE}")
        
    ds_ref = gdal.Open(str(REPROJECTED_DEM_FILE))
    if ds_ref is None:
        raise RuntimeError(f"Cannot open reference DEM: {REPROJECTED_DEM_FILE}")
        
    cols = ds_ref.RasterXSize
    rows = ds_ref.RasterYSize
    geotransform = ds_ref.GetGeoTransform()
    projection = ds_ref.GetProjection()
    ds_ref = None
    
    print(f"Reference DEM size: {rows}x{cols}")

    # 2. 读取 SWW 文件
    print(f"Reading SWW file: {SWW_FILE}...")
    ds = Dataset(SWW_FILE, mode='r')
    
    # 获取变量
    stage_var = get_variable(ds, ['stage'])
    elev_var = get_variable(ds, ['elevation'])
    
    if stage_var is None or elev_var is None:
        raise RuntimeError("SWW file must contain 'stage' and 'elevation'.")
        
    # 检查维度 (Time, Nodes)
    # 注意：确保 SWW 是基于顶点的 (Centroids 模式需要不同的处理，但通常 SWW 包含顶点数据)
    # 如果只有 centroids 数据，需要先插值到顶点，或者你的 mesh_utils 必须保证顶点顺序一致
    
    # 你的 mesh_utils 生成的 points 是 (rows * cols) 个
    # 顺序是 row-major: (0,0), (0,1)... (1,0)...
    
    n_nodes = stage_var.shape[1]
    expected_nodes = rows * cols
    
    if n_nodes < expected_nodes:
         # 严格检查：如果节点数不对，可能是边界处理导致顶点数变化？
         print(f"Warning: SWW nodes ({n_nodes}) != DEM pixels ({expected_nodes})")
         raise RuntimeError("SWW has fewer nodes than DEM pixels. Cannot reshape directly.")
    
    stage_arr = np.array(stage_var) # (Time, Nodes)
    elev_arr = np.array(elev_var)   # (Time, Nodes) or (Nodes,) or (1, Nodes)

    # 3. 构建有效节点掩膜 (Active Mask)
    # 只输出属于三角形(elements)的顶点数据，剔除孤立点
    volumes_var = get_variable(ds, ['volumes', 'elements', 'connectivity'])
    if volumes_var is not None:
        print("Building active mask from elements...")
        volumes = np.array(volumes_var)
        valid_mask_flat = np.zeros(n_nodes, dtype=bool)
        # 将所有出现在三角形中的顶点标记为 True
        valid_node_indices = np.unique(volumes)
        # 确保索引在有效范围内
        valid_node_indices = valid_node_indices[valid_node_indices < n_nodes]
        valid_mask_flat[valid_node_indices] = True
    else:
        print("Warning: Connectivity data not found. All points will be exported.")
        valid_mask_flat = np.ones(n_nodes, dtype=bool)

    #以此处理恒定高程的情况
    if elev_arr.ndim == 1:
        elev_arr = elev_arr[np.newaxis, :]
    elif elev_arr.shape[0] == 1:
        pass # OK
    
    ntime = stage_arr.shape[0]
    print(f"Processing {ntime} frames...")

    driver = gdal.GetDriverByName("GTiff")

    for t_idx in tqdm(range(ntime), desc="Exporting"):
        # 计算水深
        s = stage_arr[t_idx, :]
        z = elev_arr[t_idx, :] if elev_arr.shape[0] > 1 else elev_arr[0, :]
        depth = s - z
        depth = np.maximum(depth, 0.0)
        
        # 核心逻辑：直接 Reshape
        # 注意：mesh_utils 是按 row 优先生成的 (r in rows, c in cols)，这完全符合 numpy/gdal 的存储顺序
        try:
            # 尝试只取前 rows*cols 个（万一有多余的？）
            if depth.size >= rows * cols:
                grid = depth[:rows*cols].reshape((rows, cols))
                
                # 应用 Mask: 将非 Element 区域设为 NoData
                mask_grid = valid_mask_flat[:rows*cols].reshape((rows, cols))
                grid[~mask_grid] = -9999.0
            else:
                 raise ValueError("Not enough points")
        except ValueError:
            print(f"\nError reshaping: depth size {depth.size} -> ({rows}, {cols})")
            raise

        out_path = os.path.join(OUT_DIR, f"{QUANTITY}_{t_idx:04d}.tif")
        
        out_ds = driver.Create(out_path, cols, rows, 1, gdal.GDT_Float32)
        out_ds.SetGeoTransform(geotransform)
        out_ds.SetProjection(projection)
        
        band = out_ds.GetRasterBand(1)
        band.WriteArray(grid)
        band.SetNoDataValue(-9999.0)
        
        # 显式计算统计信息 (False 表示不使用近似值，进行精确计算)
        band.ComputeStatistics(False)
        
        out_ds.FlushCache()
        out_ds = None

    ds.close()
    print("✅ Export completed.")

if __name__ == "__main__":
    main()
