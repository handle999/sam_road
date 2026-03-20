import matplotlib
# 【关键修改】强制使用非交互式后端，防止在无界面服务器上报错
matplotlib.use('Agg') 
import matplotlib.pyplot as plt
import re
import os
import time

# ================= 配置区域 =================
# LOG_FILE_PATH = './train_logs/sam_road_xian_cityscale.txt'
# SAVE_IMG_PATH = './train_logs/sam_road_xian_cityscale_live_loss.png'  # 图表保存路径
# LOG_FILE_PATH = './train_logs/sam_road_cityscale.txt'
# SAVE_IMG_PATH = './train_logs/sam_road_cityscale_live_loss.png'  # 图表保存路径
LOG_FILE_PATH = './train_logs/sam_road_spacenet.txt'
SAVE_IMG_PATH = './train_logs/sam_road_spacenet_live_loss.png'  # 图表保存路径
# LOG_FILE_PATH = './train_logs/sam_road_spacenet_4c_update.txt'
# SAVE_IMG_PATH = './train_logs/sam_road_spacenet_4c_update_live_loss.png'  # 图表保存路径
UPDATE_INTERVAL = 10  # 每隔 10 秒刷新并保存一次图片
# ============================================

train_pattern = re.compile(
    r'Epoch\s+(\d+).*?(\d+)/(\d+)\s*\[.*?train_mask_loss=([0-9.]+),\s*train_topo_loss=([0-9.]+),\s*train_loss=([0-9.]+)'
)

val_pattern = re.compile(
    r'val_mask_loss=([0-9.]+),\s*'
    r'val_topo_loss=([0-9.]+),\s*'
    r'val_loss=([0-9.]+)'
)

train_steps = []
train_mask_losses = []
train_topo_losses = []
train_total_losses = []

val_records = []

last_file_pos = 0

# 初始化图表
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
fig.tight_layout(pad=5.0)

line_t_mask, = ax1.plot([], [], label='Train Mask', color='blue', alpha=0.7)
line_t_topo, = ax1.plot([], [], label='Train Topo', color='green', alpha=0.7)
line_t_total, = ax1.plot([], [], label='Train Total', color='red', alpha=0.9)
ax1.set_title("Live Training Loss")
ax1.set_xlabel("Global Step")
ax1.set_ylabel("Loss")
ax1.legend()
ax1.grid(True, linestyle='--', alpha=0.6)

line_v_mask, = ax2.plot([], [], marker='o', label='Val Mask', color='blue')
line_v_topo, = ax2.plot([], [], marker='o', label='Val Topo', color='green')
line_v_total, = ax2.plot([], [], marker='o', label='Val Total', color='red')
ax2.set_title("Validation Loss (Updated at End of Epoch)")
ax2.set_xlabel("Epoch")
ax2.set_ylabel("Loss")
ax2.legend()
ax2.grid(True, linestyle='--', alpha=0.6)

def update_and_save():
    global last_file_pos
    
    if not os.path.exists(LOG_FILE_PATH):
        print(f"[{time.strftime('%H:%M:%S')}] 等待日志文件生成: {LOG_FILE_PATH}")
        return
    
    with open(LOG_FILE_PATH, 'r', encoding='utf-8', errors='ignore') as f:
        f.seek(last_file_pos)
        chunk = f.read()
        last_file_pos = f.tell()
        
    if not chunk:
        return # 没有新内容直接返回

    lines = chunk.replace('\r', '\n').split('\n')
    updated = False
    
    for line in lines:
        if not line.strip():
            continue
            
        # 1. 解析 Train Loss
        train_match = train_pattern.search(line)
        if train_match:
            epoch = int(train_match.group(1))
            step = int(train_match.group(2))
            total_steps = int(train_match.group(3))
            
            t_mask = float(train_match.group(4))
            t_topo = float(train_match.group(5))
            t_loss = float(train_match.group(6))
            
            global_step = epoch * total_steps + step
            
            if not train_steps or global_step > train_steps[-1]:
                train_steps.append(global_step)
                train_mask_losses.append(t_mask)
                train_topo_losses.append(t_topo)
                train_total_losses.append(t_loss)
            else:
                train_mask_losses[-1] = t_mask
                train_topo_losses[-1] = t_topo
                train_total_losses[-1] = t_loss
            updated = True
            
        # 2. 解析 Validation Loss
        val_match = val_pattern.search(line)
        if val_match:
            v_mask = float(val_match.group(1))
            v_topo = float(val_match.group(2))
            v_loss = float(val_match.group(3))
            current_val = (v_mask, v_topo, v_loss)
            
            if not val_records or current_val != val_records[-1]:
                val_records.append(current_val)
                updated = True

    if updated:
        # 更新图表数据
        line_t_mask.set_data(train_steps, train_mask_losses)
        line_t_topo.set_data(train_steps, train_topo_losses)
        line_t_total.set_data(train_steps, train_total_losses)
        ax1.relim()
        ax1.autoscale_view()

        if val_records:
            epochs = list(range(len(val_records))) 
            v_masks = [r[0] for r in val_records]
            v_topos = [r[1] for r in val_records]
            v_totals = [r[2] for r in val_records]
            
            line_v_mask.set_data(epochs, v_masks)
            line_v_topo.set_data(epochs, v_topos)
            line_v_total.set_data(epochs, v_totals)
            ax2.relim()
            ax2.autoscale_view()
            ax2.xaxis.set_major_locator(plt.MaxNLocator(integer=True))

        # 保存为图片
        fig.savefig(SAVE_IMG_PATH, dpi=150, bbox_inches='tight')
        print(f"[{time.strftime('%H:%M:%S')}] 图表已更新并保存至 {SAVE_IMG_PATH}")

print("开始后台监控日志文件，按 Ctrl+C 停止...")
try:
    while True:
        update_and_save()
        time.sleep(UPDATE_INTERVAL)
except KeyboardInterrupt:
    print("\n监控已停止。")
