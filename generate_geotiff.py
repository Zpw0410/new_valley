#!/usr/bin/env python3
import os
import numpy as np
from netCDF4 import Dataset
from scipy.interpolate import griddata
from osgeo import gdal, osr
from tqdm import tqdm  # 进度条库

# ================= 固定路径 =================
SWW_FILE = r'd:/new_valley/DEM_Basin.sww'
OUT_DIR = r'd:/new_valley/geotiff_output'
RESOLUTION = 10.0       # 栅格分辨率
QUANTITY = 'depth'      # 改成 'depth'
CRS = 'EPSG:4326'       # 输出坐标系

# ---------------- Helpers ----------------
def get_variable(ds, candidates):
    for name in candidates:
        if name in ds.variables:
            return ds.variables[name][:]
    return None

def read_sww_basic(path):
    ds = Dataset(path, mode='r')
    info = {}
    val = get_variable(ds, ['x', 'node_x', 'longitude', 'lon'])
    info['x'] = np.array(val) if val is not None else np.array([])
    val = get_variable(ds, ['y', 'node_y', 'latitude', 'lat'])
    info['y'] = np.array(val) if val is not None else np.array([])
    val = get_variable(ds, ['volumes', 'triangles', 'elements', 'cells'])
    info['volumes'] = np.array(val, dtype=int) if val is not None else None
    val = get_variable(ds, ['stage_c', 'stage'])
    info['stage'] = np.array(val) if val is not None else None
    val = get_variable(ds, ['elevation_c', 'elevation'])
    info['elevation'] = np.array(val) if val is not None else None
    val = get_variable(ds, ['time', 'times'])
    info['time'] = np.array(val) if val is not None else None
    ds.close()
    return info

def compute_centroids(x_nodes, y_nodes, volumes):
    v0 = volumes[:, 0]
    v1 = volumes[:, 1]
    v2 = volumes[:, 2]
    xc = (x_nodes[v0] + x_nodes[v1] + x_nodes[v2]) / 3.0
    yc = (y_nodes[v0] + y_nodes[v1] + y_nodes[v2]) / 3.0
    return xc, yc

def write_geotiff_gdal(grid, out_path, xmin, ymax, pixel_size, crs_epsg='4326'):
    ny, nx = grid.shape
    driver = gdal.GetDriverByName("GTiff")
    ds = driver.Create(out_path, nx, ny, 1, gdal.GDT_Float32)
    ds.SetGeoTransform([xmin, pixel_size, 0, ymax, 0, -pixel_size])
    srs = osr.SpatialReference()
    srs.ImportFromEPSG(int(crs_epsg.split(":")[-1]))
    ds.SetProjection(srs.ExportToWkt())
    ds.GetRasterBand(1).WriteArray(grid)
    ds.FlushCache()
    ds = None

# ---------------- Main ----------------
def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    info = read_sww_basic(SWW_FILE)

    x_nodes = info['x']
    y_nodes = info['y']
    vols = info['volumes']
    stage_var = info['stage']
    elev_var = info['elevation']
    times = info['time']

    if stage_var is None:
        raise RuntimeError("Cannot find 'stage' variable in SWW file.")
    if elev_var is None:
        raise RuntimeError("Cannot find 'elevation' variable in SWW file, cannot compute depth.")

    stage_arr = np.array(stage_var)
    if stage_arr.ndim == 1:
        stage_arr = stage_arr[np.newaxis, :]
    ntime = stage_arr.shape[0]

    n_nodes = x_nodes.size
    n_tris = vols.shape[0] if vols is not None else 0
    second_dim = stage_arr.shape[1]

    if second_dim == n_nodes:
        mode = 'node'
        px = x_nodes
        py = y_nodes
    elif second_dim == n_tris:
        if vols is None:
            raise RuntimeError("stage appears triangle-based but 'volumes' missing.")
        mode = 'triangle'
        px, py = compute_centroids(x_nodes, y_nodes, vols)
    else:
        raise RuntimeError(f"Cannot determine stage type: nodes={n_nodes}, tris={n_tris}, second_dim={second_dim}")

    xmin, xmax = px.min(), px.max()
    ymin, ymax = py.min(), py.max()
    nx = int(np.ceil((xmax - xmin)/RESOLUTION)) + 1
    ny = int(np.ceil((ymax - ymin)/RESOLUTION)) + 1
    grid_x = np.linspace(xmin, xmax, nx)
    grid_y = np.linspace(ymin, ymax, ny)
    grid_xx, grid_yy = np.meshgrid(grid_x, grid_y)

    print(f"Grid: {nx}x{ny}, {ntime} time steps.")

    # ================= 进度条 =================
    for t_idx in tqdm(range(ntime), desc="Processing timesteps"):
        vals = stage_arr[t_idx, :]

        # 计算 depth
        elev_at_t = np.array(elev_var)
        if elev_at_t.ndim == 2:
            elev_at_t = elev_at_t[t_idx, :]
        if mode == 'triangle' and elev_at_t.size == n_tris:
            interp_vals = vals - elev_at_t
        elif mode == 'node' and elev_at_t.size == n_nodes:
            interp_vals = vals - elev_at_t
        else:
            raise RuntimeError("Mismatch stage/elevation shapes for depth.")

        # 水深不能为负
        interp_vals = np.maximum(interp_vals, 0.0)

        # 插值到规则网格
        grid_q = griddata((px, py), interp_vals, (grid_xx, grid_yy), method='linear')
        nan_mask = np.isnan(grid_q)
        if np.any(nan_mask):
            grid_q[nan_mask] = griddata((px, py), interp_vals, (grid_xx, grid_yy), method='nearest')[nan_mask]
        grid_q[np.isnan(grid_q)] = 0.0
        grid_q = grid_q.astype(np.float32)

        out_path = os.path.join(OUT_DIR, f"{QUANTITY}_{t_idx:04d}.tif")
        write_geotiff_gdal(grid_q, out_path, xmin, ymax, RESOLUTION, CRS)

    print("✅ All done.")

if __name__ == "__main__":
    main()
