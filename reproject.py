from osgeo import gdal

from config import DEM_FILE, REPROJECTED_DEM_FILE

src_ds = gdal.Open(str(DEM_FILE))
if src_ds is None:
    raise FileNotFoundError(f'Could not open source DEM file: {DEM_FILE}')

dst_ds = gdal.Warp(
    str(REPROJECTED_DEM_FILE),
    src_ds,
    dstSRS='EPSG:32615',
    format='GTiff',
    resampleAlg=gdal.GRA_Bilinear,
    xRes=90,
    yRes=90
)

dst_ds = None
src_ds = None