# prepare_dataset/download_use_osm.py
import sys  
import json 
import mapdriver as md 
# import mapbox as md2
import esri as md2
import graph_ops as graphlib 
import math 
import numpy as np 
import argparse
from PIL import Image 
import pickle 
import os
import osmium
from osgeo import ogr

# ==========================================
# 模式 1：PBF 解析器 (保留你的原始逻辑)
# ==========================================
class OSMHandler(osmium.SimpleHandler):
    def __init__(self):
        super().__init__()
        self.nodes = {}      # node_id -> (lat, lon)
        self.ways = []       # list of node_id sequences

    def node(self, n):
        self.nodes[n.id] = (n.location.lat, n.location.lon)

    def way(self, w):
        if len(w.nodes) >= 2:
            self.ways.append([n.ref for n in w.nodes])


def build_osmmap_from_pbf(handler, bbox):
    nodedict = {}
    nodes = handler.nodes 
    for way in handler.ways:
        keep_way = False
        for nid in way:
            if nid in nodes:
                lat, lon = nodes[nid]
                if in_bbox(lat, lon, bbox):
                    keep_way = True
                    break

        if not keep_way:
            continue 

        for u, v in zip(way[:-1], way[1:]):
            if u not in nodes or v not in nodes:
                continue
            for a, b in [(u, v), (v, u)]:
                if a not in nodedict:
                    lat, lon = nodes[a]
                    nodedict[a] = {"lat": lat, "lon": lon, "to": {}, "from": {}}
                nodedict[a]["to"][b] = 1
    return nodedict


# ==========================================
# 模式 2：SHP 解析器 
# ==========================================
def load_global_ways_from_shp(shp_path):
    print(f"[INFO] Loading geometry from SHP: {shp_path}")
    driver = ogr.GetDriverByName("ESRI Shapefile")
    ds = driver.Open(shp_path, 0)
    layer = ds.GetLayer()
    
    global_ways = []
    for feature in layer:
        geom = feature.GetGeometryRef()
        if geom is not None:
            pts = geom.GetPoints()
            # OGR is (lon, lat), model expects (lat, lon)
            way = [(p[1], p[0]) for p in pts]
            global_ways.append(way)
    print(f"[INFO] Loaded {len(global_ways)} paths.")
    return global_ways

# ==========================================
# 通用工具
# ==========================================
def in_bbox(lat, lon, bbox):
    lat_st, lon_st, lat_ed, lon_ed = bbox
    return lat_st <= lat <= lat_ed and lon_st <= lon <= lon_ed


# ==========================================
# 主程序入口
# ==========================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Sat2Graph Dataset Generator with Trajectory Augmentation")
    parser.add_argument("--dataset_type", choices=['original', 'trajectory'], default='original', 
                        help="'original': Only Sat + OSM_GT. 'trajectory': Sat + Active_SHP (Input) -> OSM_GT.")
    parser.add_argument("--osm_pbf", required=True, help="Path to .osm.pbf file (Supervisory Signal/GT)")
    parser.add_argument("--active_shp", help="Path to active edges .shp file (Required if type='trajectory')")
    parser.add_argument("configs", nargs="+", help="Dataset JSON config files")
    args = parser.parse_args()

    if args.dataset_type == 'trajectory' and not args.active_shp:
        raise ValueError("[ERROR] --active_shp must be provided when --dataset_type is 'trajectory'")

    # --- 1. 全局数据预加载 ---
    print(f"[INFO] Parsing Full OSM PBF file for Ground Truth: {args.osm_pbf}")
    handler = OSMHandler()
    handler.apply_file(args.osm_pbf, locations=True)
    
    global_shp_ways = None
    if args.dataset_type == 'trajectory':
        global_shp_ways = load_global_ways_from_shp(args.active_shp)

    # --- 2. 加载 Configs ---
    dataset_cfg = []
    total_regions = 0 
    tid = 0 
    tn = 1 

    for name_cfg in args.configs:
        dataset_cfg_ = json.load(open(name_cfg, "r"))
        for item in dataset_cfg_:
            region_name = item["region"]
            year = item["year"]
            size = item["size"]
            
            lat_min, lon_min = item["lat_min"], item["lon_min"]        
            lat_max, lon_max = item["lat_max"], item["lon_max"]        
            
            dlat = size / 111111.0           
            dlon = size / (111111.0 * math.cos(math.radians(lat_min)))  
            lat_n = math.ceil((lat_max - lat_min) / dlat)  
            lon_n = math.ceil((lon_max - lon_min) / dlon)  
            
            dataset_cfg.append({
                "lat": lat_min, "lon": lon_min, "lat_n": lat_n, "lon_n": lon_n               
            })
            total_regions += lat_n * lon_n   

    print(f"[INFO] Total regions to process: {total_regions}")
    dataset_folder = "{}_{}_{}".format(region_name, year, size)
    folder_mapbox_cache = "cache_esri/{}/".format(region_name)
    os.makedirs(dataset_folder, exist_ok=True)
    os.makedirs(folder_mapbox_cache, exist_ok=True)

    # --- 3. 核心切片循环 ---
    c = 0
    for item in dataset_cfg:
        ilat, ilon = item["lat_n"], item["lon_n"]
        lat_origin, lon_origin = item["lat"], item["lon"]

        for i in range(ilat):
            for j in range(ilon):
                print(f"Processing region {c}/{total_regions}")
                if c % tn != tid:
                    c += 1; continue

                def norm(x, nd=7): return round(x, nd)
                lat_st = norm(lat_origin + size/111111.0 * i)
                lon_st = norm(lon_origin + size/111111.0 * j / math.cos(math.radians(lat_origin)))
                lat_ed = norm(lat_origin + size/111111.0 * (i+1))
                lon_ed = norm(lon_origin + size/111111.0 * (j+1) / math.cos(math.radians(lat_origin)))
                bbox = [lat_st, lon_st, lat_ed, lon_ed]
                zoom = 18 if abs(lat_st) < 30 else 17

                # --- 3.1 截取卫星图 (Input) ---
                img, _ = md2.GetMapInRect(lat_st, lon_st, lat_ed, lon_ed, zoom=zoom, folder=folder_mapbox_cache)
                img = Image.fromarray(img.astype(np.uint8)).resize((size, size), Image.BILINEAR)
                Image.fromarray(np.array(img)).save(f"{dataset_folder}/region_{c}_sat.png")

                # --- 3.2 构建完整 OSM 拓扑图 (Ground Truth / 监督信号) ---
                node_neighbor_gt = {}
                node_dict = build_osmmap_from_pbf(handler, bbox)
                for node_id, node_info in node_dict.items():
                    n1key = (node_info["lat"], node_info["lon"])
                    neighbors = list(node_info["to"].keys()) + list(node_info["from"].keys())
                    for nid in set(neighbors):
                        n2key = (node_dict[nid]["lat"], node_dict[nid]["lon"])
                        node_neighbor_gt = graphlib.graphInsert(node_neighbor_gt, n1key, n2key)

                # GT 后处理及保存
                node_neighbor_gt = graphlib.graphDensify(node_neighbor_gt)
                nn_gt_region = graphlib.graph2RegionCoordinate(node_neighbor_gt, [lat_st,lon_st,lat_ed,lon_ed], size)
                
                with open(f"{dataset_folder}/region_{c}_graph_gt.pickle", "wb") as f:
                    pickle.dump(nn_gt_region, f)
                graphlib.graphVis2048Segmentation(node_neighbor_gt, [lat_st,lon_st,lat_ed,lon_ed], f"{dataset_folder}/region_{c}_gt.png", size)
                
                nn_refine_gt, sample_points = graphlib.graphGroundTruthPreProcess(nn_gt_region)
                with open(f"{dataset_folder}/region_{c}_refine_gt_graph.p", "wb") as f:
                    pickle.dump(nn_refine_gt, f)
                with open(f"{dataset_folder}/region_{c}_refine_gt_graph_samplepoints.json", "w") as f:
                    json.dump(sample_points, f, indent=2)

                # --- 3.3 构建活跃轨迹图 (Model Input / 你的创新点) ---
                if args.dataset_type == 'trajectory':
                    node_neighbor_active = {}
                    for way in global_shp_ways:
                        if any(in_bbox(p[0], p[1], bbox) for p in way):
                            for k in range(len(way) - 1):
                                node_neighbor_active = graphlib.graphInsert(node_neighbor_active, way[k], way[k+1])
                                node_neighbor_active = graphlib.graphInsert(node_neighbor_active, way[k+1], way[k])
                    
                    # 对活跃轨迹进行相同的插值和坐标转换
                    node_neighbor_active = graphlib.graphDensify(node_neighbor_active)
                    nn_active_region = graphlib.graph2RegionCoordinate(node_neighbor_active, [lat_st,lon_st,lat_ed,lon_ed], size)
                    
                    # 保存为 mask 图片 (可与卫星图 concat) 以及 pickle 图结构
                    graphlib.graphVis2048Segmentation(node_neighbor_active, [lat_st,lon_st,lat_ed,lon_ed], f"{dataset_folder}/region_{c}_active.png", size)
                    with open(f"{dataset_folder}/region_{c}_active_graph.pickle", "wb") as f:
                        pickle.dump(nn_active_region, f)

                c += 1
                