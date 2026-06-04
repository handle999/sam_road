import requests
import json
from datetime import datetime


def get_osm_roads(date="2014-06-01", region="xian", output_file=None):
	"""
	从 ohsome API 获取指定日期、区域的 OSM 道路数据 (GeoJSON格式)
	支持自定义日期和区域

	参数：
	- date: str, 日期字符串，格式为 'YYYY-MM-DD'
	- region: str, 区域，可选 'xian'（默认）或自定义边界 [minlon,minlat,maxlon,maxlat]
	- output_file: str, 输出文件名，默认自动生成
	"""

	# ---- 1. 西安的默认经纬度范围 (WGS84) ----
	region_bbox = {
		"xian": [108.8, 33.8, 109.2, 34.6],  # 西安市周边
	}

	# ---- 2. 解析区域 ----
	if isinstance(region, str):
		if region not in region_bbox:
			raise ValueError(f"未知区域 '{region}'，目前仅支持 {list(region_bbox.keys())} 或自定义bbox。")
		bbox = region_bbox[region]
	elif isinstance(region, (list, tuple)) and len(region) == 4:
		bbox = region
	else:
		raise ValueError("region 参数必须是 'xian' 或 [minlon, minlat, maxlon, maxlat] 格式。")

	# ---- 3. 输出文件名 ----
	if output_file is None:
		output_file = f"roads_{region}_{date}.geojson"

	# ---- 4. 构造请求 ----
	url = "https://api.ohsome.org/v1/elements/geometry"
	form_data = {
		"bboxes": f"{bbox[0]},{bbox[1]},{bbox[2]},{bbox[3]}",
		"filter": "highway=* and type:way",
		"time": date
	}

	print(f"📡 正在请求 {date} 的 OSM 道路数据...")
	print(f"📍 区域范围: {bbox}")

	# ---- 5. 发送请求 ----
	response = requests.post(url, data=form_data)

	if response.status_code != 200:
		print(f"❌ 请求失败: {response.status_code}, {response.text}")
		return

	data = response.json()

	# ---- 6. 保存结果 ----
	with open(output_file, "w", encoding="utf-8") as f:
		json.dump(data, f, ensure_ascii=False)

	print(f"✅ 已保存为 {output_file}，可直接导入 QGIS 查看。")


# ---------------------
# 示例运行
# ---------------------
if __name__ == "__main__":
	# 示例1：获取 2015年1月1日 西安道路
	get_osm_roads(date="2025-01-01", region="xian")

# 示例2：自定义区域 (bbox)
# get_osm_roads(date="2020-01-01", region=[108.7, 33.9, 109.3, 34.7])
