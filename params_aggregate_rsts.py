import os
import csv
import json

# 配置文件路径
PARAMS_CSV = 'param_exps.csv'
RESULTS_CSV = 'param_rsts.csv'
SAVE_DIR = './save'

def aggregate_results():
    if not os.path.exists(PARAMS_CSV):
        print(f"[ERROR] Cannot find {PARAMS_CSV}. Make sure you run this in the root dir.")
        return

    # 1. 读取原始参数配置
    with open(PARAMS_CSV, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        param_fieldnames = reader.fieldnames
        rows = list(reader)

    # 2. 定义最终输出表的表头 (原参数 + 新指标)
    metric_fieldnames = ['APLS', 'TOPO_Pre', 'TOPO_Rec', 'TOPO_F1']
    final_fieldnames = param_fieldnames + metric_fieldnames

    aggregated_data = []

    print("Aggregating results...")
    for row in rows:
        exp_id = row.get('exp_id', '').strip()
        if not exp_id:
            continue

        exp_dir = os.path.join(SAVE_DIR, exp_id)
        
        # 初始化默认指标值为 N/A (未跑或出错)
        row['APLS'] = 'N/A'
        row['TOPO_Pre'] = 'N/A'
        row['TOPO_Rec'] = 'N/A'
        row['TOPO_F1'] = 'N/A'

        # ==========================================
        # 提取 APLS 指标
        # ==========================================
        apls_file = os.path.join(exp_dir, 'results/apls.json')
        if os.path.exists(apls_file):
            try:
                with open(apls_file, 'r') as jf:
                    apls_data = json.load(jf)
                    # 【注意】：请根据 apls.py 实际生成的 JSON 键名修改这里
                    # 假设 JSON 长这样: {"final_APLS": 0.7637}
                    row['APLS'] = apls_data.get('final_APLS', 'N/A') 
            except Exception as e:
                print(f"[WARN] Failed to read APLS for {exp_id}: {e}")

        # ==========================================
        # 提取 TOPO 指标 (Pre, Rec, F1)
        # ==========================================
        topo_file = os.path.join(exp_dir, 'results/topo.json')
        if os.path.exists(topo_file):
            try:
                with open(topo_file, 'r') as jf:
                    topo_data = json.load(jf)
                    mean_topo = topo_data.get('mean topo', [])

                    if isinstance(mean_topo, list) and len(mean_topo) >= 3:
                        # "mean topo" order: [F1, Precision, Recall]
                        row['TOPO_F1']  = mean_topo[0]
                        row['TOPO_Pre'] = mean_topo[1]
                        row['TOPO_Rec'] = mean_topo[2]
            except Exception as e:
                print(f"[WARN] Failed to read TOPO for {exp_id}: {e}")

        aggregated_data.append(row)

    # 3. 写入最终结果表格
    with open(RESULTS_CSV, 'w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=final_fieldnames)
        writer.writeheader()
        writer.writerows(aggregated_data)

    print(f"\n[SUCCESS] All results aggregated perfectly into '{RESULTS_CSV}'!")
    print(f"Total experiments processed: {len(aggregated_data)}")

if __name__ == "__main__":
    aggregate_results()
    