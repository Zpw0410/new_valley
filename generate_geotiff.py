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
    # ds_ref = None  # We need this later for reconstruction
    
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
    
    stage_arr = np.array(stage_var) # (Time, Nodes)
    elev_arr = np.array(elev_var)   # (Time, Nodes) or (Nodes,) or (1, Nodes)

    # 3. 处理节点映射关系 (关键修改)
    # 由于 mesh_utils 现在跳过了 NaN 点，我们需要重建 (idx -> r, c) 的关系
    # 或者重建一个全尺寸的数组，把有效数据填进去
    
    print("Reconstructing mesh mapping from reference DEM...")
    band = ds_ref.GetRasterBand(1)
    ref_elev = band.ReadAsArray()
    nodata_val = band.GetNoDataValue()
    
    # 复刻 mesh_utils 的逻辑来确定哪些点是有效的
    # 注意：必须与 mesh_utils.py 的逻辑完全一致
    if nodata_val is not None:
        is_valid = (ref_elev != nodata_val) & (~np.isnan(ref_elev))
    else:
        is_valid = ~np.isnan(ref_elev)
        
    # 计算有效点的总数，校验是否与 SWW 匹配
    valid_count = np.sum(is_valid)
    
    print(f"Valid pixels in DEM: {valid_count}")
    print(f"Nodes in SWW: {n_nodes}")
    
    if valid_count != n_nodes:
        # 如果不匹配，可能是 mesh_utils 在生成时还有额外过滤？
        # 或者 SWW 包含了 centroid 模式？
        print("Warning: Node count mismatch! attempting best-effort mapping...")
        if valid_count > n_nodes:
             raise RuntimeError("SWW has fewer nodes than valid DEM pixels. Cannot map back.")
    
    # 创建一个索引映射图 (rows, cols)，值为对应的节点索引，-1 表示无效
    node_mapping = np.full((rows, cols), -1, dtype=np.int32)
    
    # 生成 0 到 valid_count-1 的序列
    valid_indices = np.arange(valid_count, dtype=np.int32)
    
    # 只有当数量严格匹配时才能这样直接赋值
    if valid_count == n_nodes:
        node_mapping[is_valid] = valid_indices
    else:
        # 如果数量不匹配，我们需要更复杂的逻辑或报错
        # 这里假设是完全匹配的，因为是同一套流程生成的
        raise RuntimeError(f"Strict node count mismatch: DEM Valid {valid_count} vs SWW {n_nodes}")

    ntime = stage_arr.shape[0]
    print(f"Processing {ntime} frames...")
    
    # 4. 导出循环
    # -----------------
    
    driver = gdal.GetDriverByName("GTiff")

    for t_idx in tqdm(range(ntime), desc="Exporting"):
        # 计算水深
        s = stage_arr[t_idx, :]
        
        # 弹性处理 elev_arr 的维度
        if elev_arr.ndim == 1:
            z = elev_arr # (Nodes,)
        else:
            # (Time, Nodes) or (1, Nodes)
            if elev_arr.shape[0] > 1:
                 z = elev_arr[t_idx, :]
            else:
                 z = elev_arr[0, :]
                 
        depth = s - z
        
        # 过滤负水深 (数值误差)
        depth = np.maximum(depth, 0.0)
        
        # 初始化输出网格
        grid = np.full((rows, cols), -9999.0, dtype=np.float32)
        
        # 使用 node_mapping 将一维的数据映射回二维网格
        # grid[r, c] = depth[ node_mapping[r, c] ] 
        # 等价于：grid[is_valid] = depth
        
        grid[is_valid] = depth
        
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
