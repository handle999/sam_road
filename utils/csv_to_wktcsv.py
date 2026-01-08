import pandas as pd
import re


def parse_coords_to_wkt(coords_str):
	# 去掉括号和多余字符
	coords_str = coords_str.strip("[]\" ")
	# 提取经纬度对
	points = []
	for part in coords_str.split(","):
		nums = re.findall(r"[-+]?\d+\.\d+", part)
		if len(nums) >= 2:
			lon, lat = nums[:2]
			points.append(f"{lon} {lat}")
	if len(points) < 2:
		return None
	return "LINESTRING(" + ", ".join(points) + ")"


def convert_to_wkt_csv(input_csv, output_csv):
	chunksize = 10000
	with open(output_csv, "w", encoding="utf-8", newline="") as out_f:
		out_f.write("traj_id,user_id,WKT\n")
		for chunk in pd.read_csv(input_csv, header=None, chunksize=chunksize):
			chunk.columns = ["traj_id", "user_id", "coords"]
			chunk["WKT"] = chunk["coords"].apply(parse_coords_to_wkt)
			chunk[["traj_id", "user_id", "WKT"]].dropna().to_csv(
				out_f, header=False, index=False
			)


# 使用示例
convert_to_wkt_csv(
	"../dataset/xianshi_1001_1015/sample_traj_1000.csv",
	"../dataset/xianshi_1001_1015/traj_linestring_1000.csv"
)
