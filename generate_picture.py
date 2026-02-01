import rasterio
import numpy as np
import folium
from folium.plugins import HeatMap
from pyproj import Transformer
import io
from PIL import Image
import time
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service 
from webdriver_manager.chrome import ChromeDriverManager
import time
import base64
import io
import os

# 读取 GeoTIFF 数据
def read_geotiff(file_path):
    dataset = rasterio.open(file_path)
    data = dataset.read(1)
    transform = dataset.transform
    crs = dataset.crs
    bounds = dataset.bounds
    print("data shape:", data.shape)
    print("transform:", transform)
    print("crs:", crs)
    print("bounds:", bounds)
    return data, transform, crs, bounds, dataset

# 生成底图（支持 CRS 转换到 WGS84）
def generate_map(center, zoom_start=12):
    m = folium.Map(location=center, zoom_start=zoom_start, tiles='cartodbpositron')
    return m

# 将 GeoTIFF 数据转为 HeatMap 格式（经纬度）
def get_heat_data(data, dataset):
    if dataset.crs != 'EPSG:4326' and dataset.crs is not None:
        transformer = Transformer.from_crs(dataset.crs, "EPSG:4326", always_xy=True)
    else:
        transformer = None

    heat_data = []
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            x, y = dataset.xy(i, j)
            if transformer:
                lon, lat = transformer.transform(x, y)
            else:
                lon, lat = x, y
            value = data[i, j]
            if np.isfinite(value) and (dataset.nodata is None or value != dataset.nodata):
                heat_data.append([lat, lon, float(value)])
    return heat_data

# 添加热力图层
def add_heatmap(m, heat_data):
    if heat_data:
        HeatMap(heat_data, radius=8, blur=15).add_to(m)

# 绘制矩形（蓝色半透明框）
def draw_rectangle(m, rect_bounds, color='blue', fill_opacity=0.2):
    folium.Rectangle(
        bounds=[[rect_bounds[1], rect_bounds[0]], [rect_bounds[3], rect_bounds[2]]],
        color=color, fill=True, fill_opacity=fill_opacity
    ).add_to(m)

# 保存 folium 地图为高分辨率 PNG
def save_map_to_png(m, output_path, dpi=300, delay=10):
    img_data = m._to_png(delay=delay)
    img = Image.open(io.BytesIO(img_data))
    img.save(output_path, dpi=(dpi, dpi), quality=95)
    print(f"已保存图像：{output_path} @ {dpi} DPI")

def save_map_to_png_precise_crop(m, output_path, bounds_wgs84, dpi=300):
    options = Options()
    options.add_argument("--headless")
    options.add_argument("--window-size=1600,1200")
    options.add_argument("--disable-gpu")
    options.add_argument("--no-sandbox")           # 加这个在 headless 下更稳定
    options.add_argument("--disable-dev-shm-usage")

    chrome_path = r"C:\Program Files\Google\Chrome\Application\chrome.exe"
    if os.path.exists(chrome_path):
        options.binary_location = chrome_path
    # else: 如果路径不对，就让系统默认找（通常可以）

    service = Service(ChromeDriverManager().install())
    driver = webdriver.Chrome(service=service, options=options)

    tmp_html = "temp_map.html"
    m.save(tmp_html)

    file_url = "file://" + os.path.abspath(tmp_html)
    driver.get(file_url)
    time.sleep(8)  # 增加等待时间到 8-12 秒，确保瓦片和 JS 完全加载

    try:
        # 关键修改在这里
        map_container = driver.find_element("css selector", "div.folium-map")
        
        # 可选：打印出来确认找到的是正确的元素
        print("找到地图容器:", map_container.get_attribute("outerHTML")[:200])  # 只打印前200字符

        # 获取容器大小（可选参考，但其实我们主要用 JS 计算像素边界）
        # map_size = map_container.size

        # JS 脚本部分保持不变，但确保变量名一致
        min_lon, min_lat, max_lon, max_lat = bounds_wgs84

        script = f"""
        var map = Object.values(window).find(obj => obj instanceof L.Map);
        if (!map) {{ return {{error: "No Leaflet map found"}}; }}
        
        var sw = map.latLngToContainerPoint(L.latLng({min_lat}, {min_lon}));
        var ne = map.latLngToContainerPoint(L.latLng({max_lat}, {max_lon}));
        
        return {{
            left:   Math.round(Math.min(sw.x, ne.x)),
            top:    Math.round(Math.min(ne.y, sw.y)),
            right:  Math.round(Math.max(ne.x, sw.x)),
            bottom: Math.round(Math.max(sw.y, ne.y))
        }};
        """

        pixel_bounds = driver.execute_script(script)

        if "error" in pixel_bounds:
            raise RuntimeError("JS 执行失败: " + pixel_bounds["error"])

        # 全屏截图
        png_base64 = driver.get_screenshot_as_base64()
        img = Image.open(io.BytesIO(base64.b64decode(png_base64)))

        crop_box = (
            pixel_bounds["left"],
            pixel_bounds["top"],
            pixel_bounds["right"],
            pixel_bounds["bottom"]
        )

        # 防止负值或越界
        crop_box = (
            max(0, crop_box[0]),
            max(0, crop_box[1]),
            min(img.width,  crop_box[2]),
            min(img.height, crop_box[3])
        )

        cropped = img.crop(crop_box)
        cropped.save(output_path, dpi=(dpi, dpi), quality=95)
        print(f"精确裁剪保存：{output_path} (像素边界: {crop_box})")

    except Exception as e:
        print("截图/裁剪过程中出错:", str(e))
        # 可选：保存全屏图用于调试
        with open("debug_full_screenshot.png", "wb") as f:
            f.write(base64.b64decode(png_base64))
        print("已保存全屏调试图：debug_full_screenshot.png")

    finally:
        driver.quit()

# 主函数
def main(geotiff_path, dpi=300):
    data, transform, crs, bounds, dataset = read_geotiff(geotiff_path)

    # 计算矩形区域（原始坐标系）
    rect_left   = bounds.left   + (bounds.right  - bounds.left)  * 0.4
    rect_right  = bounds.left   + (bounds.right  - bounds.left)  * 0.6
    rect_bottom = bounds.bottom + (bounds.top    - bounds.bottom) * 0.4
    rect_top    = bounds.bottom + (bounds.top    - bounds.bottom) * 0.6

    # 转换为 WGS84 经纬度（folium 需要）
    if crs != 'EPSG:4326' and crs is not None:
        transformer = Transformer.from_crs(crs, "EPSG:4326", always_xy=True)
        r_left,  r_bottom  = transformer.transform(rect_left,  rect_bottom)
        r_right, r_top     = transformer.transform(rect_right, rect_top)
    else:
        r_left, r_bottom, r_right, r_top = rect_left, rect_bottom, rect_right, rect_top

    rectangle_bounds_wgs84 = [r_left, r_bottom, r_right, r_top]

    # 计算地图中心（全图中心，使用 WGS84）
    if crs != 'EPSG:4326' and crs is not None:
        min_lon, min_lat = transformer.transform(bounds.left, bounds.bottom)
        max_lon, max_lat = transformer.transform(bounds.right, bounds.top)
        center = [(min_lat + max_lat) / 2, (min_lon + max_lon) / 2]
    else:
        center = [(bounds.bottom + bounds.top) / 2, (bounds.left + bounds.right) / 2]

    # 准备热力图数据（只需计算一次）
    heat_data = get_heat_data(data, dataset)

    # ────────────────────────────────────────────────
    # 第一张图：全范围 + 矩形框标记
    # ────────────────────────────────────────────────
    # m_full = generate_map(center, zoom_start=10)  # 较小 zoom 以显示全图
    # add_heatmap(m_full, heat_data)
    # draw_rectangle(m_full, rectangle_bounds_wgs84)
    # # 可选：不缩放，让全图自然显示；或 m_full.fit_bounds(...) 如果想稍微放大
    # save_map_to_png(m_full, "full_map_with_rectangle.png", dpi=dpi, delay=8)

    # ────────────────────────────────────────────────
    # 第二张图：放大到矩形区域内部
    # ────────────────────────────────────────────────
    m_zoom = generate_map([(r_bottom + r_top)/2, (r_left + r_right)/2], zoom_start=14)
    add_heatmap(m_zoom, heat_data)
    # draw_rectangle(m_zoom, rectangle_bounds_wgs84, fill_opacity=0.1)  # 透明度降低，避免挡住内容
    # 强制缩放到矩形区域（最关键）
    m_zoom.fit_bounds([[r_bottom, r_left], [r_top, r_right]])
    
    # 替换原来的 save_map_to_png
    save_map_to_png_precise_crop(
            m_zoom,
            "zoomed_rectangle_cropped.png",
            rectangle_bounds_wgs84,   # [min_lon, min_lat, max_lon, max_lat]
            dpi=300
        )

    print("完成！生成两张图像：")
    print("1. full_map_with_rectangle.png     （全图 + 矩形标记）")
    print("2. zoomed_rectangle.png             （放大后的矩形区域）")


if __name__ == "__main__":
    geotiff_path = 'depth_1440.tif'
    main(geotiff_path, dpi=300)