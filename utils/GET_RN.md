# 1. OSM

- [Open Street Map](https://www.openstreetmap.org/)
- [Geofabrik](https://download.geofabrik.de/)
- [China](https://download.geofabrik.de/asia/china.html)
- [raw directory index](https://download.geofabrik.de/asia/china.html#)

# 2. Target Region

## Xi'an

34.20, 108.91, 34.29, 109.01

34°12'0.0000"

108°54'36.0000"

34°17'24.0000"

109°0'36.0000"

Sijie Ruan code
- [tptk](https://github.com/sjruan/tptk)
- [osm2rn](https://github.com/sjruan/osm2rn)

```python
# osm2rn osm_clip.py
python ./utils/osm2rn/osm_clip.py --input_path ./dataset/osm/china-140101.osm.pbf --output_path ./dataset/osm/xian-140101.osm.pbf --min_lat 34.20 --min_lng 108.91 --max_lat 34.29 --max_lng 109.01

python ./utils/osm2rn/osm_clip.py --input_path ./dataset/osm/china-190101.osm.pbf --output_path ./dataset/osm/xian-190101.osm.pbf --min_lat 34.20 --min_lng 108.91 --max_lat 34.29 --max_lng 109.01

python ./utils/osm2rn/osm_clip.py --input_path ./dataset/osm/china-140101.osm.pbf --output_path ./dataset/osm/xian-plus-140101.osm.pbf --min_lat 33.70 --min_lng 108.40 --max_lat 34.80 --max_lng 109.50

python ./utils/osm2rn/osm_clip.py --input_path ./dataset/osm/china-190101.osm.pbf --output_path ./dataset/osm/xian-plus-190101.osm.pbf --min_lat 33.70 --min_lng 108.40 --max_lat 34.80 --max_lng 109.50

# osm2rn osm_to_rn.py

python ./utils/osm2rn/osm_to_rn.py --input_path ./dataset/osm/xian-140101.osm.pbf --output_path ./dataset/osm/xian140101

python ./utils/osm2rn/osm_to_rn.py --input_path ./dataset/osm/xian-190101.osm.pbf --output_path ./dataset/osm/xian190101

```

## 

```shell
python network_stats.py --input_path ../xian/osm/xian-plus-190101.osm.pbf
python network_stats.py --input_path ../xian/osm/xian-plus-190101-double/edges.shp

(SAM) E:\School\2025\20250311Road\GraphBased\sam_road\utils>python network_stats.py --input_path ../xian/osm/xian-plus-190101-double/edges.shp

==================================================
📊 分析构建后的路网 Shapefile: ../xian/osm/xian-plus-190101-double/edges.shp
==================================================
E:\School\2025\20250311Road\GraphBased\sam_road\utils\network_stats.py:56: DeprecationWarning: read_shp is deprecated and will be removed in 3.0.See https://networkx.org/documentation/latest/auto_examples/index.html#geospatial.
  G = nx.read_shp(shp_path, simplify=True, strict=False)
📍 交叉口/节点数量 (# of nodes): 139301
🛣️ 有向路段/边数量 (# of edges): 264543
🔗 平均度 (Average Degree): 3.80 (每个路口平均连接的路段数)

--- 🗺️ 拓扑连通性分析 ---
🟢 弱连通分量数量 (Weakly Connected Components): 74
🔴 强连通分量数量 (Strongly Connected Components): 2175
🌐 最大连通网覆盖了 98.85% 的节点 (物理相连)
🚗 最大可达网覆盖了 97.27% 的节点 (遵循道路方向可互达)
==================================================


(SAM) E:\School\2025\20250311Road\GraphBased\sam_road\utils>python network_stats.py --input_path ../xian/osm/xian-plus-190101.osm.pbf

==================================================
📊 分析原始 OSM PBF 文件: ../xian/osm/xian-plus-190101.osm.pbf
==================================================
📍 原始节点总数 (Total Raw Nodes): 151094
🛣️ 原始路线总数 (Total Raw Ways): 24625
🚗 有效道路数量 (Valid Highway Ways): 22622

💡 提示: 在 OSM 中，一条 'Way' 通常包含多个 'Node'。
   在转换为路网图 (Shapefile) 时，一条 Way 会被切分成多段 'Edges'，
   并且双向车道会生成两条平行的 Edges。
==================================================
```
