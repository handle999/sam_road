"""
Simple example that counts the number of objects in an osm file.
Shows how to write a handler for the different types of objects.
"""
import osmium as o
import argparse
from tqdm import tqdm


class WayCounter(o.SimpleHandler):
    def __init__(self):
        super().__init__()
        self.count = 0

    def way(self, w):
        self.count += 1


class NodeRetrievingHandler(o.SimpleHandler):

    def __init__(self, min_lat, min_lng, max_lat, max_lng, total_ways):
        super(NodeRetrievingHandler, self).__init__()
        self.nodes = set()
        self.min_lat = min_lat
        self.min_lng = min_lng
        self.max_lat = max_lat
        self.max_lng = max_lng
        self.pbar = tqdm(total=total_ways, desc="Scanning ways (collect nodes)")

    def way(self, w):
        self.pbar.update(1)

        if 'highway' in w.tags:
            need_add = False
            for n in w.nodes:
                if contains(self.min_lat, self.min_lng, self.max_lat, self.max_lng, n.lat, n.lon):
                    need_add = True
                    break
            if need_add:
                for n in w.nodes:
                    self.nodes.add(n.ref)

    def close(self):
        self.pbar.close()


class HighwayRetrievingHandler(o.SimpleHandler):

    def __init__(self, min_lat, min_lng, max_lat, max_lng, nodes, writer, total_ways):
        super(HighwayRetrievingHandler, self).__init__()
        self.min_lat = min_lat
        self.min_lng = min_lng
        self.max_lat = max_lat
        self.max_lng = max_lng
        self.nodes = nodes
        self.writer = writer
        self.pbar = tqdm(total=total_ways, desc="Writing highways")

    def node(self, n):
        if n.id in self.nodes:
            self.writer.add_node(n)

    def way(self, w):
        self.pbar.update(1)

        need_add = False
        try:
            if 'highway' in w.tags:
                for n in w.nodes:
                    if contains(self.min_lat, self.min_lng, self.max_lat, self.max_lng, n.lat, n.lon):
                        need_add = True
                        break
                if need_add:
                    self.writer.add_way(w)
        except o.InvalidLocationError:
                # A location error might occur if the osm file is an extract
                # where nodes of ways near the boundary are missing.
                print("WARNING: way %d incomplete. Ignoring." % w.id)

    def close(self):
        self.pbar.close()


def contains(min_lat, min_lng, max_lat, max_lng, lat, lng):
    return min_lat <= lat < max_lat and min_lng <= lng < max_lng


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--min_lat', type=float, default=39.8451, help='the min lat of the interested region')
    parser.add_argument('--min_lng', type=float, default=116.2810, help='the min lng of the interested region')
    parser.add_argument('--max_lat', type=float, default=39.9890, help='the max lat of the interested region')
    parser.add_argument('--max_lng', type=float, default=116.4684, help='the max lng of the interested region')
    parser.add_argument('--input_path', help='the input path of the original osm data')
    parser.add_argument('--output_path', help='the output path of the clipped osm data')

    opt = parser.parse_args()
    print(opt)

    # hhy add, counter & tqdm
    counter = WayCounter()
    counter.apply_file(opt.input_path, locations=False)
    total_ways = counter.count
    print("Total ways:", total_ways)

    # go through the ways to find all relevant nodes
    nh = NodeRetrievingHandler(opt.min_lat, opt.min_lng, opt.max_lat, opt.max_lng, total_ways)
    nh.apply_file(opt.input_path, locations=True, idx='flex_mem')   # 内存索引（最稳定）
    nh.close()

    # go through the file again and write out the data
    writer = o.SimpleWriter(opt.output_path)
    hh = HighwayRetrievingHandler(opt.min_lat, opt.min_lng, opt.max_lat, opt.max_lng, nh.nodes, writer, total_ways)
    hh.apply_file(opt.input_path, locations=True, idx='flex_mem')   # 内存索引（最稳定）
    hh.close()
    writer.close()
