import math
import numpy as np
import os
from PIL import Image
from time import sleep
import urllib.request


def lonlat2TileIndex(lonlat, zoom):
    """经纬度转瓦片索引（Web Mercator 投影）"""
    n = np.exp2(zoom)
    x = int((lonlat[0] + 180) / 360 * n)
    y = int((1 - math.log(math.tan(lonlat[1] * math.pi / 180) + 1 / math.cos(lonlat[1] * math.pi / 180)) / math.pi) / 2 * n)
    return [x, y]


def lonlat2TilePos(lonlat, zoom, tile_size=256):
    """经纬度转瓦片内偏移像素位置"""
    n = np.exp2(zoom)
    fx = (lonlat[0] + 180) / 360 * n
    fy = (1 - math.log(math.tan(lonlat[1] * math.pi / 180) + 1 / math.cos(lonlat[1] * math.pi / 180)) / math.pi) / 2 * n

    ix = int(fx)
    iy = int(fy)

    dx = int((fx - ix) * tile_size)
    dy = int((fy - iy) * tile_size)
    return dx, dy


def downloadTileImage(zoom, tile_xy, outputname):
    """下载 Esri 瓦片影像"""
    url = f"https://services.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{zoom}/{tile_xy[1]}/{tile_xy[0]}"
    tmp_file = outputname + ".tmp.jpg"

    Succ = False
    retry_timeout = 10

    print(f"Downloading: {url}")
    while not Succ:
        try:
            urllib.request.urlretrieve(url, tmp_file)
            os.rename(tmp_file, outputname)
            Succ = True
        except Exception as e:
            print(f"Download failed: {e}")
            print(f"Retrying in {retry_timeout} seconds...")
            sleep(retry_timeout)
            retry_timeout = min(retry_timeout + 10, 60)

    return Succ


def GetMapInRect(min_lat, min_lon, max_lat, max_lon, folder="tile_cache/", zoom=19, tile_size=256):
    os.makedirs(folder, exist_ok=True)

    tile1 = lonlat2TileIndex([min_lon, min_lat], zoom)
    tile2 = lonlat2TileIndex([max_lon, max_lat], zoom)
    print(f"tiles: {tile1}, {tile2}")
    print(f"delta of tiles: {(tile2[0]-tile1[0])*(tile2[1]-tile1[1])}")

    x_start, y_start = tile1
    x_end, y_end = tile2

    x_range = range(min(x_start, x_end), max(x_start, x_end) + 1)
    y_range = range(min(y_start, y_end), max(y_start, y_end) + 1)

    dimx = len(x_range) * tile_size
    dimy = len(y_range) * tile_size

    img = np.zeros((dimy, dimx, 3), dtype=np.uint8)

    ok = True

    for i, x in enumerate(x_range):
        if not ok:
            break
        for j, y in enumerate(y_range):
            filename = os.path.join(folder, f"{zoom}_{x}_{y}.jpg")
            if not os.path.isfile(filename):
                Succ = downloadTileImage(zoom, [x, y], filename)
                if not Succ:
                    ok = False
                    break

            subimg = Image.open(filename).convert("RGB")
            subimg = np.array(subimg).astype(np.uint8)
            img[j * tile_size:(j + 1) * tile_size, i * tile_size:(i + 1) * tile_size, :] = subimg

    # 计算裁剪偏移
    x1, y1 = lonlat2TilePos([min_lon, max_lat], zoom, tile_size)
    x2, y2 = lonlat2TilePos([max_lon, min_lat], zoom, tile_size)

    x2 += dimx - tile_size
    y2 += dimy - tile_size

    img = img[y1:y2, x1:x2]

    return img, ok
