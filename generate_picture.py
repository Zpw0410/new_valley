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
import base64
import os

# 读取 GeoTIFF 数据
def read_geotiff(file_path):
    dataset = rasterio.open(file_path)
    data = dataset.read(1)
    transform = dataset.transform
    crs = dataset.crs
    bounds = dataset.bounds
    # 过滤-9999后计算有效深度范围
    valid_data = data[(np.isfinite(data)) & (data != -9999) & (data != dataset.nodata if dataset.nodata is not None else True)]
    print("data shape:", data.shape)
    print("transform:", transform)
    print("crs:", crs)
    print("bounds:", bounds)
    print("有效深度值范围：最小值 =", np.min(valid_data) if len(valid_data) > 0 else "无有效数据", "最大值 =", np.max(valid_data) if len(valid_data) > 0 else "无有效数据")
    return data, transform, crs, bounds, dataset

# 生成底图（支持 CRS 转换到 WGS84）
def generate_map(center, zoom_start=12):
    m = folium.Map(location=center, zoom_start=zoom_start, tiles='cartodbpositron')
    return m

# 将 GeoTIFF 数据转为 HeatMap 格式（经纬度 + 归一化深度值，过滤-9999）
# 将 GeoTIFF 数据转为 HeatMap 格式（经纬度 + 归一化深度值，过滤-9999 和 浅水 <0.2m）
def get_heat_data(data, dataset, shallow_threshold=0.2):
    if dataset.crs != 'EPSG:4326' and dataset.crs is not None:
        transformer = Transformer.from_crs(dataset.crs, "EPSG:4326", always_xy=True)
    else:
        transformer = None

    valid_values = []
    temp_data = []  # (x, y, value)

    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            x, y = dataset.xy(i, j)
            value = data[i, j]

            if (np.isfinite(value)
                and (dataset.nodata is None or value != dataset.nodata)
                and value != -9999
                and value >= shallow_threshold):           # ← 新增：过滤浅水区

                valid_values.append(value)
                temp_data.append((x, y, value))

    if not valid_values:
        print("警告：无有效深度数据（过滤后为空）")
        return []

    min_val = np.min(valid_values)
    max_val = np.max(valid_values)

    if max_val == min_val:
        max_val = min_val + 1e-6

    heat_data = []
    for x, y, value in temp_data:
        if transformer:
            lon, lat = transformer.transform(x, y)
        else:
            lon, lat = x, y

        # 归一化：0.2m → 接近0（最浅蓝），最深 → 接近1（最深蓝）
        normalized = (value - min_val) / (max_val - min_val)

        heat_data.append([lat, lon, float(normalized)])

    print(f"过滤后有效深度点数：{len(heat_data)} （已移除 < {shallow_threshold}m 的区域）")
    return heat_data

# 添加热力图层（自定义蓝到红色带）
# 添加热力图层（浅蓝 → 深蓝）
def add_heatmap(m, heat_data):
    if not heat_data:
        print("警告：无有效热力数据，跳过添加热力图")
        return

    # 自定义蓝色渐变（从很浅的蓝到深蓝）
    heatmap = HeatMap(
        heat_data,
        radius=8,          # 根据你数据分辨率可调 6~12
        blur=15,           # 模糊程度，可调 10~20
        max_zoom=18,
        gradient={
            0.0:  '#f7fbff',    # 极浅（接近白/透明感）
            0.2:  '#deebf7',
            0.4:  '#c6dbef',
            0.6:  '#9ecae1',
            0.8:  '#4292c6',
            1.0:  '#08306b'     # 最深蓝
        }
    )
    heatmap.add_to(m)

# 繪製矩形（只有半透明填充，無邊框）
def draw_rectangle(m, rect_bounds, fill_color='blue', fill_opacity=0.2):
    """
    參數:
        rect_bounds: [min_lon, min_lat, max_lon, max_lat]
        fill_color:   填充顏色（支援 hex、顏色名稱等）
        fill_opacity: 填充透明度 (0.0 ~ 1.0)
    """
    folium.Rectangle(
        bounds=[[rect_bounds[1], rect_bounds[0]],   # [min_lat, min_lon]
                [rect_bounds[3], rect_bounds[2]]],  # [max_lat, max_lon]
        color='transparent',          # 邊框顏色設為透明
        fill=True,
        fill_color=fill_color,        # 填充顏色
        fill_opacity=fill_opacity,
        weight=0,                     # 邊框粗細設為 0（最重要！）
        # 可選：line_opacity=0.0      # 有些版本也支援，但 weight=0 通常已足夠
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
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-dev-shm-usage")

    chrome_path = r"C:\Program Files\Google\Chrome\Application\chrome.exe"
    if os.path.exists(chrome_path):
        options.binary_location = chrome_path

    service = Service(ChromeDriverManager().install())
    driver = webdriver.Chrome(service=service, options=options)

    tmp_html = "temp_map.html"
    m.save(tmp_html)

    file_url = "file://" + os.path.abspath(tmp_html)
    driver.get(file_url)
    time.sleep(8)

    try:
        map_container = driver.find_element("css selector", "div.folium-map")
        print("找到地图容器:", map_container.get_attribute("outerHTML")[:200])

        min_lon, min_lat, max_lon, max_lat = bounds_wgs84
        script = f"""
        var map = Object.values(window).find(obj => obj instanceof L.Map);
        if (!map) {{ return {{error: "No Leaflet map found"}}; }}
        
        var sw = map.latLngToContainerPoint(L.latLng({min_lat}, {min_lon}));
        var ne = map.latLngToContainerPoint(L.latLng({max_lat}, {max_lon}));
        
        return {{
            left:   Math.round(Math.min(sw.x, ne.x)),
            top:    Math.round(Math.min(ne.y, sw.y)),
            right:   Math.round(Math.max(ne.x, sw.x)),
            bottom: Math.round(Math.max(sw.y, ne.y))
        }};
        """

        pixel_bounds = driver.execute_script(script)
        if "error" in pixel_bounds:
            raise RuntimeError("JS 执行失败: " + pixel_bounds["error"])

        png_base64 = driver.get_screenshot_as_base64()
        img = Image.open(io.BytesIO(base64.b64decode(png_base64)))

        crop_box = (
            pixel_bounds["left"],
            pixel_bounds["top"],
            pixel_bounds["right"],
            pixel_bounds["bottom"]
        )
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
        with open("debug_full_screenshot.png", "wb") as f:
            f.write(base64.b64decode(png_base64))
        print("已保存全屏调试图：debug_full_screenshot.png")

    finally:
        driver.quit()
        if os.path.exists(tmp_html):
            os.remove(tmp_html)  # 清理临时HTML文件

# 主函数
def main(geotiff_path, dpi=300):
    data, transform, crs, bounds, dataset = read_geotiff(geotiff_path)

    # 计算矩形区域（原始坐标系）
    rect_left   = bounds.left   + (bounds.right  - bounds.left)  * 0.4
    rect_right  = bounds.left   + (bounds.right  - bounds.left)  * 0.6
    rect_bottom = bounds.bottom + (bounds.top    - bounds.bottom) * 0.4
    rect_top    = bounds.bottom + (bounds.top    - bounds.bottom) * 0.6

    # 转换为 WGS84 经纬度
    if crs != 'EPSG:4326' and crs is not None:
        transformer = Transformer.from_crs(crs, "EPSG:4326", always_xy=True)
        r_left,  r_bottom  = transformer.transform(rect_left,  rect_bottom)
        r_right, r_top     = transformer.transform(rect_right, rect_top)
    else:
        r_left, r_bottom, r_right, r_top = rect_left, rect_bottom, rect_right, rect_top

    rectangle_bounds_wgs84 = [r_left, r_bottom, r_right, r_top]

    # 计算地图中心（全图中心，WGS84）
    if crs != 'EPSG:4326' and crs is not None:
        min_lon, min_lat = transformer.transform(bounds.left, bounds.bottom)
        max_lon, max_lat = transformer.transform(bounds.right, bounds.top)
        center = [(min_lat + max_lat) / 2, (min_lon + max_lon) / 2]
    else:
        center = [(bounds.bottom + bounds.top) / 2, (bounds.left + bounds.right) / 2]

    # 准备热力图数据（已过滤-9999）
    heat_data = get_heat_data(data, dataset)

    m_full = generate_map(center, zoom_start=10)  # 较小 zoom 以显示全图
    # add_heatmap(m_full, heat_data)
    draw_rectangle(m_full, rectangle_bounds_wgs84)
    # 可选：不缩放，让全图自然显示；或 m_full.fit_bounds(...) 如果想稍微放大
    save_map_to_png(m_full, "full_map_with_rectangle.png", dpi=dpi, delay=8)
    # 放大到矩形区域内部
    m_zoom = generate_map([(r_bottom + r_top)/2, (r_left + r_right)/2], zoom_start=14)
    add_heatmap(m_zoom, heat_data)  # 应用自定义色带的热力图
    m_zoom.fit_bounds([[r_bottom, r_left], [r_top, r_right]])
    
    # 保存精确裁剪的图像
    save_map_to_png_precise_crop(
            m_zoom,
            "zoomed_rectangle_cropped.png",
            rectangle_bounds_wgs84,
            dpi=300
        )

    print("完成！生成图像：zoomed_rectangle_cropped.png（蓝到红色带，越深越红，已过滤-9999无效值）")

if __name__ == "__main__":
    geotiff_path = 'depth_1440.tif'  # 替换为你的GeoTIFF路径
    main(geotiff_path, dpi=300)