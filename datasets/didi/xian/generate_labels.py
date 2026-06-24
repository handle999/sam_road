import os
import numpy as np
import shutil
import pickle
import networkx as nx
import cv2
import json
import argparse


# tile 边长像素。prepare_dataset 用 size=400 生成 400px tile, 这里必须一致。
# (历史 landmine: 曾经是 1024, 与 2019_400 数据集不符)
IMAGE_SIZE = 400
KEYPOINT_RADIUS = 3
ROAD_WIDTH = 3

# 路径约定 (与 dataset.py:360-363 一致):
#   region 文件:  <root>/xian_2019_400/region_{c}_refine_gt_graph.p
#   输出掩膜:     <root>/processed/keypoint_mask_{c}.png, road_mask_{c}.png
#   data_split:   <root>/data_split.json   (历史位置在 2019_400/ 上一层)
# 默认从 2019_400/ 目录运行, 故 region_dir=./xian_2019_400, output_dir=./processed,
# data_split 在上一层 2019_400/../data_split.json = ./../data_split.json
# 用 argparse 覆盖以兼容不同 cwd。

def create_directory(dir,delete=False):
    if os.path.isdir(dir) and delete:
        shutil.rmtree(dir)
    os.makedirs(dir,exist_ok=True)


def draw_points_on_image(size, points, radius):
    """
    Draws points on a square image using OpenCV.

    Parameters:
    - size: The size of the square image (width and height) in pixels.
    - points: A list of tuples, where each tuple represents the (x, y) coordinates of a point in pixel coordinates.
    - radius: The radius of the circles to be drawn for each point, in pixels.

    Returns:
    - A square image with the given points drawn as filled circles.
    """
    
    # Create a square image of the given size, initialized to zeros (black), with one channel (grayscale), and dtype uint8
    image = np.zeros((size, size), dtype=np.uint8)

    # Iterate through the list of points
    for point in points:
        # Draw each point as a filled circle on the image
        # The circle is drawn with center at 'point', radius as specified, color 255 (white), and filled (thickness=-1)
        cv2.circle(image, point, radius, 255, -1)

    return image

def draw_line_segments_on_image(size, line_segments, width):
    """
    Draws line segments on a square image using OpenCV.

    Parameters:
    - size: The size of the square image (width and height) in pixels.
    - line_segments: A list of tuples, where each tuple represents a line segment as ((x1, y1), (x2, y2)).
    - width: The width of the lines to be drawn, in pixels.

    Returns:
    - A square image with the given line segments drawn.
    """
    
    # Create a square image of the given size, initialized to zeros (black)
    # with one channel (grayscale), and dtype uint8
    image = np.zeros((size, size), dtype=np.uint8)

    # Iterate through the list of line segments
    for segment in line_segments:
        # Unpack the start and end points of the line segment
        (x1, y1), (x2, y2) = segment

        # Draw the line segment on the image
        # The line is drawn with color 255 (white) and the specified width
        cv2.line(image, (x1, y1), (x2, y2), 255, width)

    return image


def main():
    ap = argparse.ArgumentParser(description="从 refine_gt_graph 生成 road_mask / keypoint_mask")
    ap.add_argument('--root', default='.',
                    help='2019_400 目录 (region 文件直接在此). 默认当前目录, 即 cd 2019_400 后运行')
    ap.add_argument('--split', default=None,
                    help='data_split.json 路径. 默认 <root>/../data_split.json (与 2019_400 同级)')
    ap.add_argument('--image-size', type=int, default=IMAGE_SIZE, help='tile 边长像素 (默认 400, 须与 prepare_dataset 一致)')
    args = ap.parse_args()

    image_size = args.image_size
    root = os.path.abspath(args.root)
    region_dir = root                                   # region 文件直接在 2019_400/ 下 (扁平)
    output_dir = os.path.join(os.path.dirname(root), 'processed')  # processed/ 与 2019_400 同级
    split_path = args.split if args.split else os.path.join(os.path.dirname(root), 'data_split.json')

    print(f'[INFO] region_dir={region_dir}')
    print(f'[INFO] output_dir={output_dir}')
    print(f'[INFO] split_path={split_path}')
    print(f'[INFO] image_size={image_size}')

    create_directory(output_dir, delete=True)

    with open(split_path, 'r') as jf:
        data_list = json.load(jf)
        data_list = data_list['test'] + data_list['validation'] + data_list['train']

    for data_index, tile_index in enumerate(data_list):
        print(f'Processing tile {tile_index} ({data_index+1}/{len(data_list)}).')

        # Load GT Graph (节点坐标是 region 局部像素 (y_down, x), top-left origin)
        gt_graph = pickle.load(
            open(os.path.join(region_dir, f"region_{tile_index}_refine_gt_graph.p"), 'rb'))
        graph = nx.Graph()  # undirected
        for n, neis in gt_graph.items():
            for nei in neis:
                # 坐标已是 (y_down, x), 转成 cv2 的 (x, y). 不做 y 翻转 (cityscale/didi 风格, top-left)
                graph.add_edge((int(n[1]), int(n[0])), (int(nei[1]), int(nei[0])))

        # Collect key nodes (degree != 2)
        key_nodes = []
        for node, degree in graph.degree():
            if degree != 2:
                key_nodes.append(node)

        # Create key point mask
        keypoint_mask = draw_points_on_image(size=image_size, points=key_nodes, radius=KEYPOINT_RADIUS)

        # Create road mask
        road_mask = draw_line_segments_on_image(
            size=image_size, line_segments=graph.edges(), width=ROAD_WIDTH)

        cv2.imwrite(os.path.join(output_dir, f'keypoint_mask_{tile_index}.png'), keypoint_mask)
        cv2.imwrite(os.path.join(output_dir, f'road_mask_{tile_index}.png'), road_mask)

    print(f'[INFO] 完成: {len(data_list)} 个 tile 的掩膜写入 {output_dir}/')


if __name__ == '__main__':
    main()