import os
import argparse
import networkx as nx
from osgeo import ogr
import osmium as o

# 【消除 Warning 的关键】显式启用 GDAL/OGR 的异常抛出机制
ogr.UseExceptions()

class OSMCountHandler(o.SimpleHandler):
    """用于快速扫描并统计 PBF 文件的 Osmium Handler"""
    def __init__(self):
        super(OSMCountHandler, self).__init__()
        self.node_count = 0
        self.way_count = 0
        self.highway_count = 0
        # 你的业务中有效的高速/道路类型
        self.candi_highway_types = {
            'motorway', 'trunk', 'primary', 'secondary', 'tertiary', 'unclassified',
            'residential', 'motorway_link', 'trunk_link', 'primary_link', 'secondary_link',
            'tertiary_link', 'living_street', 'service', 'road'
        }

    def node(self, n):
        self.node_count += 1

    def way(self, w):
        self.way_count += 1
        if 'highway' in w.tags and w.tags['highway'] in self.candi_highway_types:
            self.highway_count += 1


def print_pbf_stats(pbf_path):
    print(f"\n{'='*50}")
    print(f"📊 分析原始 OSM PBF 文件: {pbf_path}")
    print(f"{'='*50}")
    
    handler = OSMCountHandler()
    handler.apply_file(pbf_path)
    
    print(f"📍 原始节点总数 (Total Raw Nodes): {handler.node_count}")
    print(f"🛣️ 原始路线总数 (Total Raw Ways): {handler.way_count}")
    print(f"🚗 有效道路数量 (Valid Highway Ways): {handler.highway_count}")
    print("\n💡 提示: 在 OSM 中，一条 'Way' 通常包含多个 'Node'。")
    print("   在转换为路网图 (Shapefile) 时，一条 Way 会被切分成多段 'Edges'，")
    print("   并且双向车道会生成两条平行的 Edges。")
    print(f"{'='*50}\n")


def print_shp_stats(shp_path):
    print(f"\n{'='*50}")
    print(f"📊 分析构建后的路网 Shapefile: {shp_path}")
    print(f"{'='*50}")
    try:
        # 加载路网图 (DiGraph)
        G = nx.read_shp(shp_path, simplify=True, strict=False)
        
        nodes = G.number_of_nodes()
        edges = G.number_of_edges()
        
        print(f"📍 交叉口/节点数量 (# of nodes): {nodes}")
        print(f"🛣️ 有向路段/边数量 (# of edges): {edges}")
        
        if nodes > 0:
            # 计算平均度 (度越高，说明路网越密集、路口岔路越多)
            avg_degree = sum(dict(G.degree()).values()) / nodes
            print(f"🔗 平均度 (Average Degree): {avg_degree:.2f} (每个路口平均连接的路段数)")
            
        print("\n--- 🗺️ 拓扑连通性分析 ---")
        try:
            # 计算连通分量 (强连通意味着任意节点间都可以双向到达)
            scc = nx.number_strongly_connected_components(G)
            wcc = nx.number_weakly_connected_components(G)
            print(f"🟢 弱连通分量数量 (Weakly Connected Components): {wcc}")
            print(f"🔴 强连通分量数量 (Strongly Connected Components): {scc}")
            
            # 计算最大连通子图覆盖率
            largest_wcc = max(nx.weakly_connected_components(G), key=len)
            largest_scc = max(nx.strongly_connected_components(G), key=len)
            print(f"🌐 最大连通网覆盖了 {len(largest_wcc)/nodes*100:.2f}% 的节点 (物理相连)")
            print(f"🚗 最大可达网覆盖了 {len(largest_scc)/nodes*100:.2f}% 的节点 (遵循道路方向可互达)")
        except Exception as e:
            print(f"无法计算连通性统计: {e}")
            
        print(f"{'='*50}\n")
            
    except Exception as e:
        print(f"❌ 读取 Shapefile 失败: {e}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="路网文件统计工具")
    parser.add_argument('--input_path', required=True, help='输入路径: .osm.pbf 文件, 或者 .shp 文件/文件夹')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.input_path):
        print(f"❌ 找不到文件或文件夹: {args.input_path}")
    elif args.input_path.endswith('.pbf'):
        print_pbf_stats(args.input_path)
    else:
        print_shp_stats(args.input_path)

    os._exit(0)
    
