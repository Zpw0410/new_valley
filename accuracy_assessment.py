import rasterio
import numpy as np
from rasterio.transform import from_origin
from rasterio.enums import Resampling
from rasterio import Affine

# 定义一个函数来读取 tif 文件
def read_tif(file_path):
    with rasterio.open(file_path) as src:
        data = src.read(1)  # 读取栅格数据
        transform = src.transform
        crs = src.crs
        nodata = src.nodata  # 获取无数据值
    return data, transform, crs, nodata

# 定义一个函数来写入新的 tif 文件
def write_tif(file_path, data, transform, crs, nodata):
    with rasterio.open(
        file_path, 'w',
        driver='GTiff',
        count=1,
        dtype=data.dtype,
        width=data.shape[1],
        height=data.shape[0],
        crs=crs,
        transform=transform,
        nodata=nodata
    ) as dst:
        dst.write(data, 1)  # 写入数据

# 计算差值
def compute_difference(base_data, other_data, nodata):
    # 这里假设 base_data, other_data 都是 numpy 数组
    # 我们首先将 nodata 值的地方屏蔽掉
    diff = np.where(
        (base_data != nodata) & (other_data != nodata),
        other_data - base_data,
        nodata
    )
    return diff

def main():
    # 读入三个 tif 文件
    base_file = 'base_45m.tif'  # 30m分辨率的tif
    file_30_90m = '45_90m.tif'  # 30-90m分辨率的tif
    file_90m = '90m.tif'        # 90m分辨率的tif

    # 读取文件
    base_data, transform, crs, nodata = read_tif(base_file)
    data_30_90m, _, _, _ = read_tif(file_30_90m)
    data_90m, _, _, _ = read_tif(file_90m)

    # 计算差值
    diff_30_90m = compute_difference(base_data, data_30_90m, nodata)
    diff_90m = compute_difference(base_data, data_90m, nodata)

    # 保存差值为新tif文件
    write_tif('diff_30_90m.tif', diff_30_90m, transform, crs, nodata)
    write_tif('diff_90m.tif', diff_90m, transform, crs, nodata)

if __name__ == "__main__":
    main()
