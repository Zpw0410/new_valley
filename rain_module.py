# rain_module.py
import os
import re
import numpy as np
from osgeo import gdal
from datetime import datetime, timedelta



class RainTifDriver:
    """
    高性能空间降雨驱动器（支持 WGS84 或投影坐标）

    改进：
    - 从文件名解析完整日期时间（YYYYMMDD_HHMMSS）
    - 支持传入模拟起始时间字符串（如 '2019313_010101' → 2019年第313天 01:01:01）
    - time_steps 存储每个 TIF 的绝对 Unix 时间戳（秒）
    - rate_func(t) 中的 t 是相对模拟时间，内部转换为绝对时间查询
    - 新增详细调试打印
    """

    def __init__(self, tif_dir, time_pattern=r'(\d{8})_(\d{6})', unit='mm/h'):
        """
        参数:
            tif_dir: 降雨 TIF 文件夹
            time_pattern: 文件名时间部分正则，默认匹配 YYYYMMDD_HHMMSS
            unit: 雨强单位 'mm/h' 或 'mm/s'
        """
        self.tif_dir = tif_dir
        self.unit = unit
        gdal.PushErrorHandler('CPLQuietErrorHandler')
        # -----------------------------
        # 读取 TIF 文件列表
        # -----------------------------
        all_tif_files = sorted([f for f in os.listdir(tif_dir) if f.lower().endswith('.tif')])
        if not all_tif_files:
            raise ValueError(f"No TIF files found in {tif_dir}")

        # 按文件名里的时间解析成 datetime 再排序
        def extract_datetime_from_filename(f):
            m = re.search(time_pattern, f)
            if not m:
                raise ValueError(f"Cannot parse time from filename: {f}")
            date_str = m.group(1)
            time_str = m.group(2)
            year   = int(date_str[:4])
            month  = int(date_str[4:6])
            day    = int(date_str[6:8])
            hour   = int(time_str[:2])
            minute = int(time_str[2:4])
            second = int(time_str[4:6])
            return datetime(year, month, day, hour, minute, second)

        # 排序
        self.tif_files = sorted(all_tif_files, key=extract_datetime_from_filename)

        # -----------------------------
        # 提取每个文件的绝对时间戳（Unix 秒）并打印调试信息
        # -----------------------------
        self.time_pattern = re.compile(time_pattern)
        self.time_steps = []       # Unix 时间戳数组
        self.datetimes = []        # datetime 对象（本地时间）

        print("\n=== 降雨 TIF 文件时间解析结果 ===")
        print(f"{'文件名':<50} {'本地时间 (YYYY-MM-DD HH:MM:SS)':<25} {'Unix 时间戳 (秒)':<15}")
        print("-" * 95)

        for f in self.tif_files:
            m = self.time_pattern.search(f)
            if m:
                date_str = m.group(1)   # YYYYMMDD
                time_str = m.group(2)   # HHMMSS

                year   = int(date_str[:4])
                month  = int(date_str[4:6])
                day    = int(date_str[6:8])
                hour   = int(time_str[:2])
                minute = int(time_str[2:4])
                second = int(time_str[4:6])

                dt = datetime(year, month, day, hour, minute, second)
                unix_time = int(dt.timestamp())          # UTC Unix 时间戳

                self.time_steps.append(unix_time)
                self.datetimes.append(dt)

                print(f"{f:<50} {dt.strftime('%Y-%m-%d %H:%M:%S'):<25} {unix_time:<15}")
            else:
                raise ValueError(f"Cannot parse time from filename: {f}")

        self.time_steps = np.array(self.time_steps)
        self.datetimes = np.array(self.datetimes)

        print(f"\n共读取 {len(self.tif_files)} 个时间步")
        print(f"时间范围：{self.datetimes[0]}  →  {self.datetimes[-1]}")
        print(f"Unix 时间范围：{self.time_steps[0]} → {self.time_steps[-1]}")
        print("=" * 50 + "\n")

        # -----------------------------
        # 读取第一个 TIF 获取地理信息
        # -----------------------------
        sample_path = os.path.join(tif_dir, self.tif_files[0])
        sample_ds = gdal.Open(sample_path)
        if sample_ds is None:
            raise RuntimeError(f"Cannot open sample TIF: {self.tif_files[0]}")

        gt = sample_ds.GetGeoTransform()
        self.origin_x = gt[0]
        self.pixel_w  = gt[1]
        self.origin_y = gt[3]
        self.pixel_h  = gt[5]  # 通常负值

        if abs(gt[2]) > 1e-12 or abs(gt[4]) > 1e-12:
            raise RuntimeError("TIF must be non-rotated (gt[2]==gt[4]==0)")

        self.ncols = sample_ds.RasterXSize
        self.nrows = sample_ds.RasterYSize

        # 像素中心坐标
        x_centers = self.origin_x + (np.arange(self.ncols) + 0.5) * self.pixel_w
        y_centers = self.origin_y + (np.arange(self.nrows) + 0.5) * self.pixel_h

        if y_centers[0] > y_centers[-1]:
            y_centers = y_centers[::-1]
            self.flip_y = True
        else:
            self.flip_y = False

        self.x_centers = x_centers
        self.y_centers = y_centers

        # -----------------------------
        # 读取所有雨强数据栈
        # -----------------------------
        ntime = len(self.tif_files)
        self.rain_stack = np.zeros((ntime, self.nrows, self.ncols), dtype=float)

        print(f"正在读取 {ntime} 个雨强 TIF ...")
        for i, fname in enumerate(self.tif_files):
            ds = gdal.Open(os.path.join(tif_dir, fname))
            arr = ds.GetRasterBand(1).ReadAsArray().astype(float)
            arr = np.nan_to_num(arr, nan=0.0)
            if self.flip_y:
                arr = arr[::-1, :]
            if self.unit == 'mm/h':
                arr /= 3600.0  # 转为 mm/s
            self.rain_stack[i] = arr
            ds = None
        print("所有雨强数据读取完成。\n")

    # ------------------------------------------------------------------
    # 解析起始时间（20190313_010101）
    # ------------------------------------------------------------------
    def parse_ymd_time(self, ymd_time_str):
        pattern = r'(\d{4})(\d{2})(\d{2})_(\d{6})'
        m = re.match(pattern, ymd_time_str)
        if not m:
            raise ValueError(f"Invalid start_time format: {ymd_time_str}. Expect YYYYMMDD_HHMMSS")
        
        year  = int(m.group(1))
        month = int(m.group(2))
        day   = int(m.group(3))
        hhmmss = m.group(4)
        hour   = int(hhmmss[:2])
        minute = int(hhmmss[2:4])
        second = int(hhmmss[4:6])
        return datetime(year, month, day, hour, minute, second)


    # ------------------------------------------------------------------
    # 生成 ANUGA 用的 rate 函数
    # ------------------------------------------------------------------
    def get_rate_function_for_mesh(self, points: np.ndarray, elements: np.ndarray, start_time_str=None):
        """
        返回 ANUGA 用的雨强函数 rate(t) → mm/s

        参数:
            start_time_str: 可选，模拟 t=0 对应的真实时刻，例如 "2019313_010101"
                            若不传，默认使用第一个 TIF 的时间作为 t=0
        """
        points = np.asarray(points)
        if points.shape[1] > 2:
            points = points[:, :2]

        elements = np.asarray(elements, dtype=int)

        # 计算三角形重心
        tri_centers = np.mean(points[elements], axis=1)

        # 预计算双线性权重（一次性完成，后面不再计算坐标）
        cols = np.searchsorted(self.x_centers, tri_centers[:, 0]) - 1
        rows = np.searchsorted(self.y_centers, tri_centers[:, 1]) - 1
        cols = np.clip(cols, 0, len(self.x_centers) - 2)
        rows = np.clip(rows, 0, len(self.y_centers) - 2)

        x0, x1 = self.x_centers[cols], self.x_centers[cols + 1]
        y0, y1 = self.y_centers[rows], self.y_centers[rows + 1]

        wx1 = (tri_centers[:, 0] - x0) / (x1 - x0)
        wx0 = 1 - wx1
        wy1 = (tri_centers[:, 1] - y0) / (y1 - y0)
        wy0 = 1 - wy1

        row_idx = np.stack([rows, rows, rows + 1, rows + 1], axis=1)
        col_idx = np.stack([cols, cols + 1, cols, cols + 1], axis=1)
        weights = np.stack([wx0 * wy0, wx1 * wy0, wx0 * wy1, wx1 * wy1], axis=1)

        # -----------------------------
        # 确定模拟起始时间（Unix 秒）
        # -----------------------------
        if start_time_str is None:
            start_unix = self.time_steps[0]
            print(f"未指定 start_time，使用第一个 TIF 时间作为 t=0：{self.datetimes[0]} (Unix {start_unix})")
        else:
            start_dt = self.parse_ymd_time(start_time_str)
            start_unix = int(start_dt.timestamp())
            print(f"用户指定模拟起始时间：{start_dt} (Unix {start_unix})")

        # -----------------------------
        # 构造 rate 函数
        # -----------------------------
        def rate_func(t):
            abs_t = start_unix + t                      # 转换为真实世界时间
            idx = np.searchsorted(self.time_steps, abs_t, side='right') - 1
            idx = np.clip(idx, 0, len(self.time_steps) - 1)
            stack_t = self.rain_stack[idx]
            return np.sum(stack_t[row_idx, col_idx] * weights, axis=1)

        return rate_func