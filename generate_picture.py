import rasterio
from rasterio.warp import calculate_default_transform, reproject, Resampling
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
from tqdm import tqdm

def reproject_to_wgs84(dataset, data, target_crs="EPSG:4326", resampling=Resampling.nearest):
    """
    將 raster 重投影到 WGS84 (EPSG:4326)
    回傳：新的 data (numpy array), 新的 transform, 新的 bounds (經緯度), 更新後的 profile
    """
    src_crs = dataset.crs
    if src_crs is None:
        src_crs = "EPSG:32615"  # 你的預設假設

    if src_crs == target_crs:
        print("已經是 EPSG:4326，無需重投影")
        return data, dataset.transform, dataset.bounds, dataset.profile

    # 計算目標 transform、寬高
    transform, width, height = calculate_default_transform(
        src_crs,
        target_crs,
        dataset.width,
        dataset.height,
        *dataset.bounds,
        resolution=(0.0001, 0.0001)   # ← 可調整解析度，單位是度，例如 0.0001 ≈ 10米左右
        # 如果想保持原始解析度，可以不設 resolution，讓它自動計算
    )

    # 建立目標 profile
    profile = dataset.profile.copy()
    profile.update({
        'crs': target_crs,
        'transform': transform,
        'width': width,
        'height': height,
        'nodata': dataset.nodata if dataset.nodata is not None else -9999
    })

    # 準備目標陣列
    dest_data = np.empty((height, width), dtype=data.dtype)

    # 執行重投影
    reproject(
        source=data,
        destination=dest_data,
        src_transform=dataset.transform,
        src_crs=src_crs,
        dst_transform=transform,
        dst_crs=target_crs,
        resampling=resampling,          # nearest 適合分級深度；cubic 或 bilinear 適合連續值
    )

    # 計算新邊界（經緯度）
    new_bounds = rasterio.transform.array_bounds(height, width, transform)

    print(f"重投影完成：新尺寸 {height}x{width}, 新邊界 {new_bounds}")
    return dest_data, transform, new_bounds, profile
    
# 读取 GeoTIFF 数据
def read_geotiff(file_path):
    dataset = rasterio.open(file_path)
    data = dataset.read(1)
    transform = dataset.transform
    crs = dataset.crs
    bounds = dataset.bounds
    # 过滤-9999后计算有效深度范围
    valid_data = data[(np.isfinite(data)) & (data != -9999) & (data != dataset.nodata if dataset.nodata is not None else True)]
    # 过滤-9999后计算有效深度范围
    valid_data = data[(np.isfinite(data)) & (data != -9999) & (data != dataset.nodata if dataset.nodata is not None else True)]
    print("data shape:", data.shape)
    print("transform:", transform)
    print("crs:", crs)
    print("bounds:", bounds)
    print("有效深度值范围：最小值 =", np.min(valid_data) if len(valid_data) > 0 else "无有效数据", "最大值 =", np.max(valid_data) if len(valid_data) > 0 else "无有效数据")
    print("有效深度值范围：最小值 =", np.min(valid_data) if len(valid_data) > 0 else "无有效数据", "最大值 =", np.max(valid_data) if len(valid_data) > 0 else "无有效数据")
    return data, transform, crs, bounds, dataset

# 生成底图（支持 CRS 转换到 WGS84）
def generate_map(center, zoom_start=12):
    m = folium.Map(location=center, zoom_start=zoom_start, tiles='cartodbpositron')
    return m

# 将 GeoTIFF 数据转为 HeatMap 格式（经纬度 + 归一化深度值，过滤-9999）
def get_heat_data(data, dataset):
    if dataset.crs != 'EPSG:4326' and dataset.crs is not None:
        transformer = Transformer.from_crs(dataset.crs, "EPSG:4326", always_xy=True)
    else:
        transformer = None

    # 第一步：收集所有有效深度值（过滤-9999、非数值、nodata）
    valid_values = []
    temp_data = []  # 临时存储有效坐标和原始深度值
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            x, y = dataset.xy(i, j)
            value = data[i, j]
            # 核心修改：增加 value != -9999 判断，过滤无效深度值
            if (np.isfinite(value) 
                and (dataset.nodata is None or value != dataset.nodata) 
                and value != -9999):
                valid_values.append(value)
                temp_data.append((x, y, value))

    # 若无有效数据，直接返回空列表
    if not valid_values:
        print("警告：无有效深度数据（所有值为-9999或无效值）")
        return []

    # 第二步：归一化深度值到 0-1 区间（0=最浅，1=最深）
    min_val = np.min(valid_values)
    max_val = np.max(valid_values)
    # 避免除零（所有深度值相同）
    if max_val == min_val:
        max_val = min_val + 1e-6

    heat_data = []
    for x, y, value in temp_data:
        if transformer:
            lon, lat = transformer.transform(x, y)
        else:
            lon, lat = x, y
        # 归一化：越深值越接近1，越浅越接近0
        normalized_val = (value - min_val) / (max_val - min_val)
        heat_data.append([lat, lon, float(normalized_val)])

    return heat_data

# 添加热力图层（自定义蓝到红色带）
def add_heatmap(m, heat_data):
    if heat_data:
        # 核心修改：自定义色带 gradient，0=蓝色（浅），1=深红色（深）
        # 可根据需求调整中间过渡色的位置和颜色
        heatmap = HeatMap(
            heat_data,
            radius=8,        # 热力点半径
            blur=15,         # 模糊程度
            gradient={       # 自定义色带：蓝→青→黄→橙→红→深红
                0.0: 'blue',
                0.2: 'cyan',
                0.4: 'yellow',
                0.6: 'orange',
                0.8: 'red',
                1.0: 'darkred'
            }
        )
        heatmap.add_to(m)
    else:
        print("警告：无有效热力数据，跳过添加热力图")

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
    ).add_to(m)

# 保存 folium 地图为高分辨率 PNG（保留HTML，优化瓦片加载）
def save_folium_as_png_chrome(
    m,
    output_path,
    dpi=300,
    max_wait=20,  # 最大等待时间（秒）
    bounds_wgs84=None,
    window_size=(1600, 1200),
    keep_temp_html=True  # 新增：是否保留临时HTML文件（方便调试）
):
    options = Options()
    options.add_argument("--headless=new")  # 新版headless模式，兼容性更好
    options.add_argument(f"--window-size={window_size[0]},{window_size[1]}")
    options.add_argument("--disable-gpu")
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-dev-shm-usage")
    # 新增：强制加载图片/瓦片，解决headless模式下瓦片不显示的问题
    options.add_argument("--enable-images")
    options.add_argument("--disable-image-loading=false")
    options.add_argument("--disable-cache")
    options.add_argument("--user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36")

    chrome_path = r"C:\Program Files\Google\Chrome\Application\chrome.exe"
    if os.path.exists(chrome_path):
        options.binary_location = chrome_path

    service = Service(ChromeDriverManager().install())
    driver = webdriver.Chrome(service=service, options=options)

    tmp_html = "temp_full_map.html"  # 全图单独命名，避免和裁剪图冲突
    m.save(tmp_html)
    print(f"已生成临时HTML文件：{os.path.abspath(tmp_html)}")

    file_url = "file://" + os.path.abspath(tmp_html)
    driver.get(file_url)

    png_base64 = ""
    try:
        # 新增：智能轮询等待瓦片加载完成（替代固定sleep）
        print("正在等待地图瓦片加载完成...")
        tile_loaded = False
        for _ in range(max_wait):
            # JS脚本：检查Leaflet地图瓦片是否加载完成
            check_script = """
            var map = Object.values(window).find(obj => obj instanceof L.Map);
            if (!map) return false;
            // 遍历所有图层，检查瓦片加载状态
            var allTilesLoaded = true;
            for (var layerId in map._layers) {
                var layer = map._layers[layerId];
                if (layer._tilesToLoad && layer._tilesToLoad > 0) {
                    allTilesLoaded = false;
                    break;
                }
            }
            return allTilesLoaded;
            """
            tile_loaded = driver.execute_script(check_script)
            if tile_loaded:
                print("瓦片加载完成！")
                break
            time.sleep(1)

        if not tile_loaded:
            print(f"警告：超过{max_wait}秒，瓦片仍未完全加载（可能是网络问题）")

        if bounds_wgs84 is not None:
            # 精确裁剪模式
            min_lon, min_lat, max_lon, max_lat = bounds_wgs84
            crop_script = f"""
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
            pixel_bounds = driver.execute_script(crop_script)
            if "error" in pixel_bounds:
                raise RuntimeError("JS 执行失败: " + pixel_bounds["error"])

            png_base64 = driver.get_screenshot_as_base64()
            img = Image.open(io.BytesIO(base64.b64decode(png_base64)))
            crop_box = (
                max(0, pixel_bounds["left"]),
                max(0, pixel_bounds["top"]),
                min(img.width,  pixel_bounds["right"]),
                min(img.height, pixel_bounds["bottom"])
            )
            cropped = img.crop(crop_box)
            cropped.save(output_path, dpi=(dpi, dpi), quality=95)
            print(f"精确裁剪保存：{output_path}")
        else:
            # 全屏保存（无裁剪）
            png_base64 = driver.get_screenshot_as_base64()
            img = Image.open(io.BytesIO(base64.b64decode(png_base64)))
            img.save(output_path, dpi=(dpi, dpi), quality=95)
            print(f"全图保存（无裁剪）：{output_path}")

    except Exception as e:
        print("截图过程中出错:", str(e))
        if 'png_base64' in locals() and png_base64:
            with open("debug_full_screenshot.png", "wb") as f:
                f.write(base64.b64decode(png_base64))
            print("已保存全屏调试图：debug_full_screenshot.png")
    finally:
        driver.quit()
        # 新增：根据keep_temp_html决定是否删除临时HTML
        if not keep_temp_html and os.path.exists(tmp_html):
            os.remove(tmp_html)
            print(f"已删除临时HTML文件：{tmp_html}")
        else:
            print(f"保留临时HTML文件供调试：{os.path.abspath(tmp_html)}")

# 保存精确裁剪的图像（保留HTML，优化瓦片加载）
def save_map_to_png_precise_crop(m, output_path, bounds_wgs84, dpi=300, keep_temp_html=True):
    options = Options()
    options.add_argument("--headless=new")
    options.add_argument("--window-size=1600,1200")
    options.add_argument("--disable-gpu")
    options.add_argument("--no-sandbox")
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-dev-shm-usage")
    # 新增：强制加载图片/瓦片
    options.add_argument("--enable-images")
    options.add_argument("--disable-image-loading=false")
    options.add_argument("--disable-cache")

    chrome_path = r"C:\Program Files\Google\Chrome\Application\chrome.exe"
    if os.path.exists(chrome_path):
        options.binary_location = chrome_path

    service = Service(ChromeDriverManager().install())
    driver = webdriver.Chrome(service=service, options=options)

    tmp_html = "temp_zoom_map.html"  # 裁剪图单独命名HTML
    m.save(tmp_html)
    print(f"已生成临时HTML文件：{os.path.abspath(tmp_html)}")

    file_url = "file://" + os.path.abspath(tmp_html)
    driver.get(file_url)
    time.sleep(8)

    try:
        map_container = driver.find_element("css selector", "div.folium-map")
        print("找到地图容器:", map_container.get_attribute("outerHTML")[:200])

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
        print("已保存裁剪调试图：debug_zoom_screenshot.png")

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
    add_heatmap(m_full, heat_data)
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