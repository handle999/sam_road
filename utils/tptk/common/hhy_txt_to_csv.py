import os
import csv
from .trajectory import parse_traj_file

def process_folder(input_folder, output_folder):
    # 1. 创建输出文件夹
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
        print(f"创建文件夹: {output_folder}")

    # 2. 遍历输入文件夹
    for filename in os.listdir(input_folder):
        if filename.endswith(".txt"):
            input_path = os.path.join(input_folder, filename)
            # 根据 QGIS 习惯，CSV 是最通用的交换格式
            output_filename = filename.replace(".txt", ".csv")
            output_path = os.path.join(output_folder, output_filename)

            print(f"正在转换: {filename} -> {output_filename}")

            try:
                # 3. 调用 trajectory.py 中的解析函数
                # 注意：由于你的数据包含路段ID等信息，必须指定 traj_type='mm'
                trajs = parse_traj_file(input_path, traj_type='mm')

                # 4. 写入 CSV 文件
                with open(output_path, 'w', encoding='utf-8', newline='') as f:
                    writer = csv.writer(f)
                    # 写入表头：轨迹ID，对象ID，开始时间，结束时间，WKT几何体
                    writer.writerow(['tid', 'oid', 'start_time', 'end_time', 'wkt'])
                    
                    for traj in trajs:
                        writer.writerow([
                            traj.tid,
                            traj.oid,
                            traj.get_start_time().strftime('%Y-%m-%d %H:%M:%S'),
                            traj.get_end_time().strftime('%Y-%m-%d %H:%M:%S'),
                            traj.to_wkt()
                        ])
            except Exception as e:
                print(f"处理文件 {filename} 时出错: {e}")

if __name__ == "__main__":
    # 配置你的路径
    INPUT_DIR = r'E:\School\2025\20251022RSTraj\dataset\clean\trajs\didi_gaia\xian'   # 存放原始 txt 的文件夹
    OUTPUT_DIR = r'E:\School\2025\20251022RSTraj\dataset\clean\vis\didi_gaia\xian' # 转换后的 csv 文件夹
    
    process_folder(INPUT_DIR, OUTPUT_DIR)
    print("\n所有文件处理完毕！")
