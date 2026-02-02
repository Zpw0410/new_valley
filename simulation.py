#!/usr/bin/env python3
# run_anuga_from_dem_fast.py
import os
import sys

if sys.platform == 'darwin':
    os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'  # 解决 macOS 下的 OpenMP 冲突问题

import time
import anuga
import shutil
import numpy as np
from anuga import Domain
from rain_module import RainTifDriver
from config import REPROJECTED_DEM_FILE
from mesh_utils import generate_mesh_from_dem
from flow_module import load_and_project_stations, create_inflow_regions, create_inlet_operators
from datetime import datetime, timedelta
import re
from mesh_build import build_mesh_from_ns_ne

SIMULATION_IN_PARALLEL = True  # 设置为 True 可启用并行计算

# -----------------------------
# 用户参数
# -----------------------------
output_dir = 'anuga_output'
os.makedirs(output_dir, exist_ok=True)

final_time = 86400 * 10  # 模拟总时间，单位秒（这里为 10 天）
yieldstep = 600
mannings_n = 0.03
print_interval = 60  # 每 60s 打印一次信息
start_time="20190310_010000"  # 模拟起始时间字符串，格式 YYYYMMDD_HHMMSS
ns_path = "ns.txt"
ne_path = "ne.txt"

if os.path.exists(output_dir):
    try:
        shutil.rmtree(output_dir)
    except OSError:
        pass # 可能文件被占用
os.makedirs(output_dir, exist_ok=True)
if os.path.exists('DEM_Basin.sww'):
    os.remove('DEM_Basin.sww')


def parse_ymd_time(time_str):
    m = re.match(r'(\d{4})(\d{2})(\d{2})_(\d{6})', time_str)
    if not m:
        raise ValueError(f"Invalid time format: {time_str}")
    year  = int(m.group(1))
    month = int(m.group(2))
    day   = int(m.group(3))
    hhmmss = m.group(4)
    hour   = int(hhmmss[:2])
    minute = int(hhmmss[2:4])
    second = int(hhmmss[4:6])
    return datetime(year, month, day, hour, minute, second)

# -----------------------------
# 1 生成网格
# -----------------------------
# points, points_lonlat, elements, boundary, elev_points = generate_mesh_from_dem(str(REPROJECTED_DEM_FILE), output_dir=output_dir, ply_pre=True)
# print(f'网格顶点数: {points.shape[0]}, 三角形数: {elements.shape[0]}')
coords, tris, elevs ,boundary, tri_types= build_mesh_from_ns_ne(ns_path, ne_path)

if coords.ndim == 3 and coords.shape[2] == 1:
    coords = coords.reshape(-1, 2)
if tris.ndim == 3 and tris.shape[2] == 1:
    tris = tris.reshape(-1, 3)

# -----------------------------
# 2 创建 Domain
# -----------------------------
print("创建 ANUGA Domain...")
domain:Domain = anuga.Domain(coords.tolist(), tris.tolist())
domain.set_name('DEM_Basin')
domain.set_quantity('elevation', elevs)
domain.set_quantity('friction', mannings_n)
domain.set_quantity('stage', expression='elevation')

if SIMULATION_IN_PARALLEL:
    domain.set_omp_num_threads(os.cpu_count())

domain.set_boundary({
    # 'left': anuga.Reflective_boundary(domain),
    # 'right': anuga.Reflective_boundary(domain),
    # 'top': anuga.Reflective_boundary(domain),
    # 'bottom': anuga.Reflective_boundary(domain),
    'exterior': anuga.Reflective_boundary(domain),
})

# -----------------------------
# 3 降雨算子
# -----------------------------
rain_driver = RainTifDriver(tif_dir='rain_32615')
rain_rate = rain_driver.get_rate_function_for_mesh(coords, tris,start_time_str=start_time)
rain = anuga.Rainfall(domain, rate=rain_rate, default_rate=0.0)
domain.forcing_terms.append(rain)
print("降雨算子已添加到 Domain")

# -----------------------------
# 5 入流算子
# -----------------------------
stations, station_ids = load_and_project_stations()
regions = create_inflow_regions(domain, stations, station_ids, radius=90.0)  # 调整 radius
inlet_ops = create_inlet_operators(domain, regions, station_ids, start_time=start_time)

print("入流算子已添加到 Domain")

sim_start_dt = parse_ymd_time(start_time)
print(f"模拟起始真实时间：{sim_start_dt}")

# -----------------------------
# 6 演化
# -----------------------------
print("开始模拟...")
# start_time = datetime(2019, 3, 10, 1, 0)  # 可选，指定模拟起始时间
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
        current_dt = sim_start_dt + timedelta(seconds=t)
        print(
            f"[t={t:.1f}s | {current_dt.strftime('%Y-%m-%d %H:%M:%S')}] "
            f"步数: {step_count}, 耗时: {elapsed:.1f}s, "
            f"stage_min={stage_min:.3f}, stage_max={stage_max:.3f}, stage_mean={stage_mean:.3f}"
        )

end_time = time.time()
print(f"模拟完成，总耗时: {end_time - start_time:.1f}s")

# -----------------------------
# 5 保存结果
# -----------------------------
out_sww = os.path.join(output_dir, 'DEM_Basin.sww')
domain.sww_merge(out_sww)
print("完成，输出：", out_sww)
