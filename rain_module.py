# rain_module.py
import os
import re
import numpy as np
from osgeo import gdal

class RainTifDriver:
    """
    高性能空间降雨驱动器（支持 WGS84 或投影坐标）
    思路：
    - 网格 points 使用真实坐标（经纬度或投影米）
    - TIF 的像素中心也用真实坐标
    - 在初始化 get_rate_function_for_mesh 时一次性计算每个三角形对应的栅格索引和双线性权重
    - 演化过程中直接使用索引和权重计算，无需再用真实坐标匹配
    """

    def __init__(self, tif_dir, time_pattern=r'(\d{8})_(\d{6})', unit='mm/h'):
        self.tif_dir = tif_dir
        self.unit = unit

        # -----------------------------
        # 读取 TIF 文件
        # -----------------------------
        self.tif_files = sorted([f for f in os.listdir(tif_dir) if f.lower().endswith('.tif')])
        if not self.tif_files:
            raise ValueError(f"No TIF files found in {tif_dir}")

        # -----------------------------
        # 提取时间戳（秒）
        # -----------------------------
        self.time_pattern = re.compile(time_pattern)
        self.time_steps = []
        for f in self.tif_files:
            m = self.time_pattern.search(f)
            if m:
                hhmmss = m.group(2)
                seconds = int(hhmmss[:2]) * 3600 + int(hhmmss[2:4]) * 60 + int(hhmmss[4:])
                self.time_steps.append(seconds)
            else:
                raise ValueError(f"Cannot parse time from filename: {f}")
        self.time_steps = np.array(self.time_steps)

        # -----------------------------
        # 读取第一个 TIF 获取尺寸与仿射
        # -----------------------------
        sample_ds = gdal.Open(os.path.join(tif_dir, self.tif_files[0]))
        if sample_ds is None:
            raise RuntimeError("Cannot open sample TIF file.")
        gt = sample_ds.GetGeoTransform()
        self.origin_x = gt[0]
        self.pixel_w = gt[1]
        self.origin_y = gt[3]
        self.pixel_h = gt[5]  # 通常为负

        if abs(gt[2]) > 1e-12 or abs(gt[4]) > 1e-12:
            raise RuntimeError("TIF must be non-rotated (gt[2]==gt[4]==0)")

        self.ncols = sample_ds.RasterXSize
        self.nrows = sample_ds.RasterYSize

        # 计算像素中心真实坐标
        x_centers = self.origin_x + (np.arange(self.ncols) + 0.5) * self.pixel_w
        y_centers = self.origin_y + (np.arange(self.nrows) + 0.5) * self.pixel_h

        # 保证 y_centers 升序
        if y_centers[0] > y_centers[-1]:
            y_centers = y_centers[::-1]
            self.flip_y = True
        else:
            self.flip_y = False

        self.x_centers = x_centers
        self.y_centers = y_centers

        # -----------------------------
        # 读取所有 TIF 栈
        # -----------------------------
        ntime = len(self.tif_files)
        self.rain_stack = np.zeros((ntime, self.nrows, self.ncols), dtype=float)
        for i, fname in enumerate(self.tif_files):
            ds = gdal.Open(os.path.join(tif_dir, fname))
            if ds is None:
                raise RuntimeError(f"Cannot open {fname}")
            arr = ds.GetRasterBand(1).ReadAsArray().astype(float)
            arr = np.nan_to_num(arr, nan=0.0)
            if self.flip_y:
                arr = arr[::-1, :]
            if self.unit == 'mm/s':
                arr *= 3600.0
            self.rain_stack[i] = arr
            ds = None

    def get_rate_function_for_mesh(self, points: np.ndarray, elements: np.ndarray):
        """
        返回 ANUGA 可用的 rate 函数
        - points: 网格点坐标 (N,2) 或 (N,3) WGS84/投影
        - elements: 三角形顶点索引 (M,3)
        """

        points = np.asarray(points)
        if points.shape[1] > 2:
            points = points[:, :2]

        elements = np.asarray(elements, dtype=int)
        M = len(elements)

        # -----------------------------
        # 1️⃣ 计算三角形重心
        # -----------------------------
        tri_centers = np.mean(points[elements], axis=1)  # (M,2)

        # -----------------------------
        # 2️⃣ 预计算每个三角形对应 TIF 栅格索引和双线性权重
        # -----------------------------
        x_centers = self.x_centers
        y_centers = self.y_centers

        # col/row索引
        cols = np.searchsorted(x_centers, tri_centers[:,0]) - 1
        rows = np.searchsorted(y_centers, tri_centers[:,1]) - 1
        cols = np.clip(cols, 0, len(x_centers)-2)
        rows = np.clip(rows, 0, len(y_centers)-2)

        x0 = x_centers[cols]
        x1 = x_centers[cols+1]
        y0 = y_centers[rows]
        y1 = y_centers[rows+1]

        wx1 = (tri_centers[:,0] - x0)/(x1 - x0)
        wx0 = 1 - wx1
        wy1 = (tri_centers[:,1] - y0)/(y1 - y0)
        wy0 = 1 - wy1

        # 向量化存储索引和权重
        row_idx = np.stack([rows, rows, rows+1, rows+1], axis=1)  # (M,4)
        col_idx = np.stack([cols, cols+1, cols, cols+1], axis=1)
        w = np.stack([wx0*wy0, wx1*wy0, wx0*wy1, wx1*wy1], axis=1)

        # -----------------------------
        # 3️⃣ 构造向量化 rate 函数
        # -----------------------------
        def rate_func(t):
            idx = np.searchsorted(self.time_steps, t, side='right') - 1
            idx = np.clip(idx, 0, len(self.time_steps)-1)
            stack_t = self.rain_stack[idx]

            # 向量化提取雨强
            tri_vals = np.sum(stack_t[row_idx, col_idx] * w, axis=1)
            return tri_vals

        return rate_func
