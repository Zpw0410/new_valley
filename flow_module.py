import anuga
import pandas as pd
import numpy as np
from pyproj import Proj, Transformer
from datetime import datetime
import io  # 添加 io 导入

# 单位转换：cfs → m³/s
CFS_TO_M3S = 0.028316846846592

# 步骤1: 读取 upstream_bci.csv 并投影坐标（修复 pyproj 语法）
def load_and_project_stations(csv_file='./upstream_bci.csv'):
    df = pd.read_csv(csv_file, dtype={'station_id': str})  # 强制 station_id 为字符串，保留前导零
    stations = {}
    
    # 使用 Transformer（推荐方式，避免 axis order 问题）
    transformer = Transformer.from_crs("EPSG:4326", "EPSG:32615", always_xy=True)
    
    for _, row in df.iterrows():
        station_id = row['station_id']  # 现在是 str，如 '06601200'
        lon, lat = row['lon'], row['lat']
        # 投影：lon, lat → x, y (UTM easting, northing)
        x, y = transformer.transform(lon, lat)
        stations[station_id] = {
            'lon_lat': (lon, lat),
            'projected': (x, y)
        }
    return stations, df['station_id'].tolist()

# 步骤2: 生成 Region（圆形区域，radius 可根据网格大小调整，例如 10–50 m）
def create_inflow_regions(domain, stations, station_ids, radius=20.0):
    regions = {}
    for station_id in station_ids:
        x, y = stations[station_id]['projected']
        region = anuga.Region(domain, center=(x, y), radius=radius)
        regions[station_id] = region
    return regions

# 步骤3: 读取流量数据，生成 Q(t) 函数（改进读取逻辑）
def load_flow_data(station_id, start_time=None):
    file_path = f"{station_id}.txt"  # station_id 是 str，带前导零 → '06601200.txt'
    
    # 读取所有行
    with open(file_path, 'r') as f:
        lines = f.readlines()
    
    # 找到数据起始：跳过所有 # 行
    data_start = 0
    for i, line in enumerate(lines):
        if not line.startswith('#'):
            data_start = i
            break
    
    # 跳过表头行和格式描述行（+2）
    data_start += 2
    
    # 剩余行为数据
    data_lines = lines[data_start:]
    
    # 使用 StringIO 读取数据
    df = pd.read_csv(
        io.StringIO(''.join(data_lines)),
        sep='\t',
        names=['agency_cd', 'site_no', 'datetime', 'tz_cd', 'discharge_cfs', 'cd'],
        parse_dates=['datetime'],
        on_bad_lines='skip'  # 防止坏行崩溃
    )
    
    # 清洗：只保留有效数据行，转换 discharge_cfs 为 float
    df = df.dropna(subset=['discharge_cfs'])
    df['discharge_cfs'] = pd.to_numeric(df['discharge_cfs'], errors='coerce')
    df = df.dropna(subset=['discharge_cfs'])  # 再次删除转换失败的行
    
    # 转换为 m³/s
    df['discharge_m3s'] = df['discharge_cfs'] * CFS_TO_M3S
    
    # 设置时间零点
    if start_time is None:
        start_time = df['datetime'].min()
    
    df['t_seconds'] = (df['datetime'] - start_time).dt.total_seconds()
    
    times = df['t_seconds'].values
    flows = df['discharge_m3s'].values
    
    # 插值函数（超出范围保持首尾值）
    def Q(t):
        if len(flows) == 0:
            return 0.0
        return np.interp(t, times, flows, left=flows[0], right=flows[-1])
    
    return Q

# 步骤4: 创建多个 Inlet_operator（自动在 evolve 中生效）
def create_inlet_operators(domain, regions, station_ids, start_time=None):
    operators = []
    for station_id in station_ids:
        Q_func = load_flow_data(station_id, start_time)
        region = regions[station_id]
        op = anuga.Inlet_operator(
            domain,
            region,
            Q=Q_func,
            verbose=False
        )
        operators.append(op)
    return operators

# ---------------- 使用示例（放在你的主脚本中） ----------------
# stations, station_ids = load_and_project_stations()
# regions = create_inflow_regions(domain, stations, station_ids, radius=20.0)
# start_time = datetime(2019, 3, 10, 0, 0)  # 根据你的模拟起始时间调整，或保持 None
# inlet_ops = create_inlet_operators(domain, regions, station_ids, start_time=start_time)
#
# # 然后直接 evolve，Inlet_operator 会自动工作
# for t in domain.evolve(yieldstep=300.0, duration=86400):
#     domain.print_timestepping_statistics()