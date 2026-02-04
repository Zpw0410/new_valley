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

# 将 GeoTIFF 数据转为 HeatMap 格式（经纬度 + 归一化深度值，过滤-9999 和 浅水 <0.2m）
def get_heat_data(data, dataset, shallow_threshold=0.2):
    # 统一处理坐标系转换（包括crs为None的情况）
    src_crs = "EPSG:32615" if dataset.crs is None else dataset.crs
    if src_crs != 'EPSG:4326':
        transformer = Transformer.from_crs(src_crs, "EPSG:4326", always_xy=True)
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
            # 0.0:  '#f7fbff',    # 极浅（接近白/透明感）
            # 0.2:  '#deebf7',
            0.0:  '#c6dbef',
            0.4:  '#9ecae1',
            0.7:  '#4292c6',
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

    # 智能等待瓦片加载
    print("正在等待裁剪图瓦片加载完成...")
    for _ in range(15):
        check_script = """
        var map = Object.values(window).find(obj => obj instanceof L.Map);
        if (!map) return true;
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
        if driver.execute_script(check_script):
            break
        time.sleep(1)

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
        with open("debug_zoom_screenshot.png", "wb") as f:
            f.write(base64.b64decode(png_base64))
        print("已保存裁剪调试图：debug_zoom_screenshot.png")

    finally:
        driver.quit()
        # 新增：保留临时HTML供调试
        if not keep_temp_html and os.path.exists(tmp_html):
            os.remove(tmp_html)
            print(f"已删除临时HTML文件：{tmp_html}")
        else:
            print(f"保留临时HTML文件供调试：{os.path.abspath(tmp_html)}")

# --------------------------
# 新增：定义离散分级规则（匹配论文图的色带）
# --------------------------
# 区间(最小值, 最大值) → 对应颜色（浅蓝到深蓝）
depth_classes = [
    (0.2, 1.0, '#c6dbef'),   # 浅蓝
    (1.0, 2.0, '#9ecae1'),   # 中浅蓝
    (2.0, 3.0, '#6baed6'),   # 中蓝
    (3.0, 4.0, '#4292c6'),   # 中深蓝
    (4.0, float('inf'), '#08306b')  # 最深蓝
]

# --------------------------
# 新增：生成离散分级的栅格图像（无效区域透明）
# --------------------------
def create_discrete_raster(data, shallow_threshold=0.2, nodata_value=-9999):
    """
    將深度數據轉為離散分級的 RGBA PNG (無效區域完全透明)
    """
    height, width = data.shape
    img = np.zeros((height, width, 4), dtype=np.uint8)

    total_pixels = height * width
    with tqdm(total=total_pixels, desc="生成離散分級圖層 (像素處理)", unit="px") as pbar:
        for i in range(height):
            for j in range(width):
                value = data[i, j]
                if (np.isfinite(value) and 
                    value != nodata_value and 
                    value >= shallow_threshold):
                    
                    for min_val, max_val, color in depth_classes:
                        if min_val <= value < max_val:
                            r = int(color.lstrip('#')[0:2], 16)
                            g = int(color.lstrip('#')[2:4], 16)
                            b = int(color.lstrip('#')[4:6], 16)
                            img[i, j] = (r, g, b, 255)
                            break
                pbar.update(1)

    # 轉 PNG base64
    img_buffer = io.BytesIO()
    Image.fromarray(img).save(img_buffer, format='PNG')
    img_buffer.seek(0)
    return base64.b64encode(img_buffer.getvalue()).decode('utf-8')


# --------------------------
# 替换：新增离散分级覆盖层（替代原热力图）
# --------------------------
def add_discrete_overlay(m, img_base64, bounds_wgs84):
    """
    添加离散分级的图像覆盖层（不透明、边缘分明）
    """
    folium.raster_layers.ImageOverlay(
        image=f'data:image/png;base64,{img_base64}',
        bounds=bounds_wgs84,  # 图像的地理范围（WGS84）
        opacity=1.0,  # 完全不透明
        interactive=False,
        cross_origin=False,
        zindex=1  # 确保在底图之上
    ).add_to(m)


# 主函数
def main(geotiff_path, dpi=300):
    data, transform, crs, bounds, dataset = read_geotiff(geotiff_path)

    print("開始重投影到 EPSG:4326 ...")
    data_4326, _, bounds_4326, _ = reproject_to_wgs84(
        dataset, data,
        target_crs="EPSG:4326",
        resampling=Resampling.nearest
    )

    min_lon, min_lat, max_lon, max_lat = bounds_4326

    # 改用更清楚的變數名稱
    full_width  = max_lon - min_lon
    full_height = max_lat - min_lat

    # 矩形區域計算（中心 20% 寬度，向左偏移一個寬度）
    rect_width   = full_width * 0.2
    rect_height  = full_height * 0.2

    rect_center_x = min_lon + full_width * 0.5
    rect_center_y = min_lat + full_height * 0.5

    rect_left   = rect_center_x - rect_width * 1.5   # 向左偏移一個 rect_width
    rect_right  = rect_center_x - rect_width * 0.5
    rect_bottom = rect_center_y - rect_height / 2
    rect_top    = rect_center_y + rect_height / 2

    rectangle_bounds_wgs84 = [rect_left, rect_bottom, rect_right, rect_top]
    full_bounds_wgs84 = [[min_lat, min_lon], [max_lat, max_lon]]

    center = [(min_lat + max_lat) / 2, (min_lon + max_lon) / 2]

    print("開始生成離散分級圖層...")
    img_base64 = create_discrete_raster(
        data_4326,
        shallow_threshold=0.2,
        nodata_value=dataset.nodata if dataset.nodata is not None else -9999
    )

    # 全圖
    m_full = generate_map(center, zoom_start=10)
    add_discrete_overlay(m_full, img_base64, full_bounds_wgs84)
    draw_rectangle(m_full, rectangle_bounds_wgs84)
    m_full.fit_bounds(full_bounds_wgs84)

    save_folium_as_png_chrome(
        m_full, "full_map_blue.png",
        dpi=dpi, max_wait=25, bounds_wgs84=None, keep_temp_html=True
    )

    # 放大區域
    zoom_center = [(rect_bottom + rect_top)/2, (rect_left + rect_right)/2]
    m_zoom = generate_map(zoom_center, zoom_start=14)
    add_discrete_overlay(m_zoom, img_base64, full_bounds_wgs84)
    m_zoom.fit_bounds([[rect_bottom, rect_left], [rect_top, rect_right]])

    save_map_to_png_precise_crop(
        m_zoom, "zoomed_rectangle_cropped.png",
        rectangle_bounds_wgs84, dpi=300, keep_temp_html=True
    )

    print("\n完成！生成檔案：")
    print("  full_map_blue.png          → 全圖")
    print("  zoomed_rectangle_cropped.png → 放大矩形區域裁切圖")
    print("  temp_full_map.html / temp_zoom_map.html → 供檢查")


if __name__ == "__main__":
    geotiff_path = './selected_pictures/depth_0476.tif'
    main(geotiff_path, dpi=300)