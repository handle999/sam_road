import csv
import os
import subprocess
import time

CSV_FILE = 'param_exps.csv'
SAVE_DIR = './save'

def run_experiments():
    if not os.path.exists(CSV_FILE):
        print(f"[ERROR] Cannot find {CSV_FILE}. Please check the path.")
        return

    # 确保基础输出目录存在
    os.makedirs(SAVE_DIR, exist_ok=True)

    with open(CSV_FILE, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        
        for row in reader:
            exp_id = row['exp_id'].strip()
            if not exp_id:
                continue  # 跳过空行
            
            # ==========================================
            # [状态机机制]: 检查当前实验是否已经跑完
            # 如果存在 inference_time.txt，直接跳过
            # ==========================================
            success_flag = os.path.join(SAVE_DIR, exp_id, 'inference_time.txt')
            if os.path.exists(success_flag):
                print(f"[SKIP] Experiment {exp_id} already finished. Skipping to next.")
                continue
            
            print(f"\n" + "="*50)
            print(f"[START] Launching Experiment: {exp_id}")
            print(f"="*50)
            
            # 基础命令 (假设你使用最佳权重的路径)
            cmd = [
                "python", "inferencer_copy.py",
                "--checkpoint", "./checkpoints/samroad_4c_update_spacenet/epoch=9-step=13230.ckpt",  # !!! 请替换为你的实际 checkpoint 路径 !!!
                "--config", row['config'],
                "--exp_id", exp_id,
                "--task", row['task']
            ]
            
            # 将 CSV 中的配置全部映射到命令行参数
            # 注意：这里的 -- 参数名必须与你在 inferencer_copy.py argparse 里定义的一致
            param_mapping = {
                'dataset': '--dataset',
                'edge': '--edge',
                'ratio': '--ratio',
                'road_thresh': '--road_thresh',
                'itsc_thresh': '--itsc_thresh',
                'topo_thresh': '--topo_thresh',
                'road_nms': '--road_nms',
                'itsc_nms': '--itsc_nms',
                'nbr_radius': '--nbr_radius',
                'max_nbr_q': '--max_nbr_q'
            }
            
            for csv_key, arg_flag in param_mapping.items():
                val = row.get(csv_key, '').strip()
                if val:  # 如果 CSV 里该项有值
                    cmd.extend([arg_flag, val])
            
            # 打印将要执行的完整命令，方便 debug
            print(f"[CMD] {' '.join(cmd)}")
            
            # ==========================================
            # 执行子进程，阻塞等待
            # ==========================================
            start_t = time.time()
            try:
                # check=True 表示如果脚本崩溃(exit code != 0)，会抛出异常
                subprocess.run(cmd, check=True)
                cost_t = time.time() - start_t
                print(f"[SUCCESS] {exp_id} completed in {cost_t:.2f} seconds.")
            except subprocess.CalledProcessError as e:
                print(f"\n[ERROR] Experiment {exp_id} failed with exit code {e.returncode}.")
                print("[WARN] Continuing to the next experiment in 3 seconds...")
                time.sleep(3)
                continue # 一个实验挂了，不影响下一个实验继续跑
            except KeyboardInterrupt:
                print("\n[STOP] Process interrupted by user. Exiting pipeline.")
                break

if __name__ == "__main__":
    run_experiments()
    