# -*- coding: utf-8 -*-
from pathlib import Path
from osgeo import gdal
import os

# ====================== 配置部分 ======================
# 输入目录：原始带坐标系的雨量 tif 文件所在文件夹
INPUT_DIR = Path("data_for_omaha_hydrology_simu_March_2019/rain/dist_rain")

# 输出目录：重投影后的文件保存位置（这里设为 rain 目录本身）
OUTPUT_DIR = Path("rain_32615")

# 目标坐标系
TARGET_SRS = "EPSG:32615"

# 目标分辨率（米）
TARGET_RES = 90.0
# =======================================================

def reproject_tif(src_path: Path, dst_path: Path):
    """对单个 tif 文件进行重投影"""
    src_ds = gdal.Open(str(src_path))
    if src_ds is None:
        print(f"无法打开文件: {src_path}")
        return False

    try:
        # 执行 Warp 重投影
        gdal.Warp(
            str(dst_path),
            src_ds,
            dstSRS=TARGET_SRS,
            format="GTiff",
            resampleAlg=gdal.GRA_Bilinear,
            xRes=TARGET_RES,
            yRes=TARGET_RES,
            # 可选：如果想覆盖已有文件，可加 overwrite=True（默认会报错）
            # options=["OVERWRITE=YES"]
        )
        print(f"已完成重投影: {src_path.name} → {dst_path.name}")
        return True
    except Exception as e:
        print(f"重投影失败 {src_path.name}: {e}")
        return False
    finally:
        src_ds = None  # 释放数据集


def main():
    # 确保输出目录存在
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # 查找所有 .tif / .tiff 文件
    tif_files = list(INPUT_DIR.glob("*.tif")) + list(INPUT_DIR.glob("*.tiff"))

    if not tif_files:
        print(f"在 {INPUT_DIR} 中没有找到任何 .tif / .tiff 文件")
        return

    print(f"找到 {len(tif_files)} 个 tif 文件，开始批量重投影...\n")

    success_count = 0
    for src_path in tif_files:
        # 保持文件名完全一致，输出到 rain 目录
        dst_filename = src_path.name
        dst_path = OUTPUT_DIR / dst_filename

        # 如果目标文件已存在，可选择跳过或覆盖
        if dst_path.exists():
            print(f"目标文件已存在，跳过: {dst_filename}")
            continue
            # 如果想强制覆盖，可注释上面两行，取消下面注释
            # print(f"目标文件已存在，将覆盖: {dst_filename}")

        success = reproject_tif(src_path, dst_path)
        if success:
            success_count += 1

    print("\n" + "="*50)
    print(f"处理完成：{success_count}/{len(tif_files)} 个文件成功重投影")
    print(f"输出目录：{OUTPUT_DIR.resolve()}")


if __name__ == "__main__":
    main()