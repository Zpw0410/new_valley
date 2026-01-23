#!/usr/bin/env python3
# run_anuga_from_dem_fast.py
import os
import numpy as np
from osgeo import gdal
import os
import sys
import time
import anuga
from anuga import Domain
from mesh_utils import generate_mesh_from_dem
from rain_module import RainTifDriver

SIMULATION_IN_PARALLEL = False  # 设置为 True 可启用并行计算

# -----------------------------
# 用户参数
# -----------------------------
dem_file = './dembnk_wgs84.tif'
output_dir = 'anuga_output'
os.makedirs(output_dir, exist_ok=True)

rain_intensity = 0.005  # m/s
final_time = 3600
yieldstep = 300
mannings_n = 0.03
print_interval = 60  # 每 60s 打印一次信息

# -----------------------------
# 1️⃣ 打开 DEM
# -----------------------------
ds = gdal.Open(dem_file)
if ds is None:
    print("ERROR: 无法打开 DEM", file=sys.stderr)
    sys.exit(1)

band = ds.GetRasterBand(1)
elevation = band.ReadAsArray().astype(np.float32)

nodata = band.GetNoDataValue()
if nodata is not None:
    elevation = np.where(elevation == nodata, np.nan, elevation)

gt = ds.GetGeoTransform()
nrows, ncols = elevation.shape

dx_deg = gt[1]
dy_deg = abs(gt[5])
yllcorner_deg = gt[3]

# -----------------------------
# 2️⃣ 经纬度 → 米
# -----------------------------
lat_center = yllcorner_deg - (nrows * dy_deg) / 2
dx = dx_deg * 111320 * np.cos(np.deg2rad(lat_center))
dy = dy_deg * 110540

print(f"DEM size: {nrows} x {ncols}")
print(f"dx={dx:.2f} m, dy={dy:.2f} m")

# -----------------------------
# 3️⃣ 生成网格
# -----------------------------
points, points_lonlat, elements, boundary, elev_points, dx, dy = generate_mesh_from_dem(dem_file, output_dir=output_dir, ply_pre=True)
print(f"网格顶点数: {points.shape[0]}, 三角形数: {elements.shape[0]}")

# -----------------------------
# 4️⃣ 创建 Domain
# -----------------------------
print("创建 ANUGA Domain...")
domain = Domain(points, elements, boundary)
domain.set_name('DEM_Basin')
domain.set_quantity('elevation', elev_points)
domain.set_quantity('friction', mannings_n)
domain.set_quantity('stage', expression='elevation')

if SIMULATION_IN_PARALLEL:
    domain.set_omp_num_threads(os.cpu_count())
    

domain.set_boundary({
    'left': anuga.Reflective_boundary(domain),
    'right': anuga.Reflective_boundary(domain),
    'top': anuga.Reflective_boundary(domain),
    'bottom': anuga.Reflective_boundary(domain),
    'exterior': anuga.Reflective_boundary(domain),
})

# -----------------------------
# 5️⃣ 降雨算子
# -----------------------------
rain_driver = RainTifDriver(tif_dir='./313')
rain_rate = rain_driver.get_rate_function_for_mesh(points, elements)
rain = anuga.Rainfall(domain, rate=rain_rate, default_rate=0.0)
domain.forcing_terms.append(rain)
print("降雨算子已添加到 Domain")

# -----------------------------
# 6️⃣ 演化
# -----------------------------
print("开始模拟...")
start_time = time.time()
step_count = 0

for t in domain.evolve(yieldstep=yieldstep, finaltime=final_time):
    step_count += 1
    if t % print_interval == 0 or t >= final_time:
        elapsed = time.time() - start_time
        # 获取当前最小/最大水位和平均水位作为演化信息
        stage_min = np.min(domain.quantities['stage'].centroid_values)
        stage_max = np.max(domain.quantities['stage'].centroid_values)
        stage_mean = np.mean(domain.quantities['stage'].centroid_values)
        print(f"[t={t:.1f}s] 步数: {step_count}, 耗时: {elapsed:.1f}s, stage_min={stage_min:.3f}, stage_max={stage_max:.3f}, stage_mean={stage_mean:.3f}")

end_time = time.time()
print(f"模拟完成，总耗时: {end_time - start_time:.1f}s")

# -----------------------------
# 7️⃣ 保存结果
# -----------------------------
out_sww = os.path.join(output_dir, 'DEM_Basin.sww')
domain.sww_merge(out_sww)
print("完成，输出：", out_sww)
