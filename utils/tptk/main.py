# a tutorial example based on T-Drive dataset
from .common.road_network import load_rn_shp    # hhy 2026/01/08, 修改为相对导入, tptk all site-package
from .common.trajectory import Trajectory, store_traj_file, parse_traj_file    # hhy 2026/01/08, 修改为相对导入, tptk all site-package
from .common.trajectory import STPoint    # hhy 2026/01/08, 修改为相对导入, tptk all site-package
from .noise_filtering import STFilter, HeuristicFilter    # hhy 2026/01/08, 修改为相对导入, tptk all site-package
from .segmentation import TimeIntervalSegmentation, StayPointSegmentation    # hhy 2026/01/08, 修改为相对导入, tptk all site-package
from .map_matching.hmm.hmm_map_matcher import TIHMMMapMatcher    # hhy 2026/01/08, 修改为相对导入, tptk all site-package
from .common.mbr import MBR    # hhy 2026/01/08, 修改为相对导入, tptk all site-package
from datetime import datetime
import os
from tqdm import tqdm
import argparse
from .statistics import statistics    # hhy 2026/01/08, 修改为相对导入, tptk all site-package


def parse_tdrive(filename, tdrive_root_dir):
    oid = filename.replace('.txt', '')
    with open(os.path.join(tdrive_root_dir, filename), 'r') as f:
        pt_list = []
        for line in f.readlines():
            attrs = line.strip('\n').split(',')
            lat = float(attrs[3])
            lng = float(attrs[2])
            time = datetime.strptime(attrs[1], '%Y-%m-%d %H:%M:%S')
            pt_list.append(STPoint(lat, lng, time))
    if len(pt_list) > 1:
        return Trajectory(oid, 0, pt_list)
    else:
        return None


def do_clean(raw_traj, filters, segmentations):
    clean_traj = raw_traj
    for filter in filters:
        clean_traj = filter.filter(clean_traj)
        if clean_traj is None:
            return []
    clean_traj_list = [clean_traj]
    for seg in segmentations:
        tmp_clean_traj_list = []
        for clean_traj in clean_traj_list:
            segment_trajs = seg.segment(clean_traj)
            tmp_clean_traj_list.extend(segment_trajs)
        clean_traj_list = tmp_clean_traj_list
    return clean_traj_list


def clean_tdrive(tdrive_root_dir, clean_traj_dir):
    start_time = datetime(2008, 2, 2)
    end_time = datetime(2008, 2, 9)
    mbr = MBR(39.8451, 116.2810, 39.9890, 116.4684)
    st_filter = STFilter(mbr, start_time, end_time)
    heuristic_filter = HeuristicFilter(max_speed=35)
    filters = [st_filter, heuristic_filter]
    ti_seg = TimeIntervalSegmentation(max_time_interval_min=6)
    sp_seg = StayPointSegmentation(dist_thresh_meter=100, max_stay_time_min=15)
    segs = [ti_seg, sp_seg]
    for filename in tqdm(os.listdir(tdrive_root_dir)):
        raw_traj = parse_tdrive(filename, tdrive_root_dir)
        if raw_traj is None:
            continue
        clean_trajs = do_clean(raw_traj, filters, segs)
        if len(clean_trajs) > 0:
            store_traj_file(clean_trajs, os.path.join(clean_traj_dir, filename))


def mm_tdrive(clean_traj_dir, mm_traj_dir, rn_path):
    rn = load_rn_shp(rn_path, is_directed=True)
    map_matcher = TIHMMMapMatcher(rn)
    for filename in tqdm(os.listdir(clean_traj_dir)):
        clean_trajs = parse_traj_file(os.path.join(clean_traj_dir, filename))
        mm_trajs = [map_matcher.match(clean_traj) for clean_traj in clean_trajs]
        store_traj_file(mm_trajs, os.path.join(mm_traj_dir, filename), traj_type='mm')


def process_mm_segmentation(input_dir, output_dir):
    # 分割MM轨迹的函数， hhy add 2026/01/08
    # 配置分割器：时间；驻留点
    ti_seg = TimeIntervalSegmentation(max_time_interval_min=5)
    # 如果需要，也可以开启驻留点分割（注意：这会删除驻留期间的点！）
    sp_seg = StayPointSegmentation(dist_thresh_meter=50, max_stay_time_min=3)
    
    segmenters = [ti_seg, sp_seg] # 如果要用驻留分割，改成 [ti_seg, sp_seg]

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    print(f"Start segmenting MM trajectories from {input_dir}...")
    
    for filename in tqdm(os.listdir(input_dir)):
        if not filename.endswith('.txt'):
            continue
            
        file_path = os.path.join(input_dir, filename)
        save_path = os.path.join(output_dir, filename)
        
        try:
            # 【关键点1】使用 traj_type='mm' 读取，保留 data['candi_pt'] 信息
            trajs = parse_traj_file(file_path, traj_type='mm')
        except Exception as e:
            print(f"Skipping {filename}: {e}")
            continue

        final_traj_list = []
        
        # 对读取到的每一条轨迹进行分割
        for traj in trajs:
            # 当前批次初始化
            current_batch = [traj]
            
            # 依次应用所有分割器
            for seg_tool in segmenters:
                next_batch = []
                for t in current_batch:
                    # 调用现有的 segmentation.py 逻辑
                    result = seg_tool.segment(t)
                    if result:
                        next_batch.extend(result)
                current_batch = next_batch
            
            final_traj_list.extend(current_batch)

        # 【关键点2】使用 traj_type='mm' 保存，将匹配信息写回文件
        if len(final_traj_list) > 0:
            store_traj_file(final_traj_list, save_path, traj_type='mm')
            
    print("Segmentation done.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--tdrive_root_dir', help='the directory of the TDrive dataset')
    parser.add_argument('--clean_traj_dir', help='the directory of the cleaned trajectories')
    parser.add_argument('--rn_path', help='the road network data path generated by osm2rn')
    parser.add_argument('--mm_traj_dir', help='the directory of the map-matched trajectories')
    parser.add_argument('--segment_output_dir', help='directory to store segmented MM trajectories')    # seg, hhy add 2026/01/08
    parser.add_argument('--phase', help='the preprocessing phase [clean,mm,stat]')

    opt = parser.parse_args()
    print(opt)

    if opt.phase == 'clean':
        clean_tdrive(opt.tdrive_root_dir, opt.clean_traj_dir)
    elif opt.phase == 'mm':
        mm_tdrive(opt.clean_traj_dir, opt.mm_traj_dir, opt.rn_path)
    elif opt.phase == 'stat':
        statistics(opt.clean_traj_dir)
    elif opt.phase == 'segment_mm':
        process_mm_segmentation(opt.mm_traj_dir, opt.segment_output_dir)    # hhy add 2026/01/08
    else:
        raise Exception('unknown phase')
