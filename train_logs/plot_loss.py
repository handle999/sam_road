import re
import pandas as pd
import plotly.express as px

def parse_and_accumulate_log(filepath):
    # pattern = re.compile(
    #     r"Epoch (\d+).*?(\d+)/\d+.*?train_mask_loss=(.*?), train_topo_loss=(.*?), train_contrastive_loss=(.*?), train_loss=(.*?)(?:, val_mask_loss=(.*?), val_topo_loss=(.*?), val_loss=(.*?))?\]"
    # )
    pattern = re.compile(
        r"Epoch (\d+).*?(\d+)/\d+.*?train_mask_loss=(.*?), train_topo_loss=(.*?), train_loss=(.*?)(?:, val_mask_loss=(.*?), val_topo_loss=(.*?), val_loss=(.*?))?\]"
    )
    data = []

    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            match = pattern.search(line)
            if match:
                epoch = int(match.group(1))
                step = int(match.group(2))
                train_mask_loss = float(match.group(3))
                train_topo_loss = float(match.group(4))
                # train_contrastive_loss = float(match.group(5))
                train_loss = float(match.group(5)) if match.group(5) else None
                val_mask_loss = float(match.group(6)) if match.group(6) else None
                val_topo_loss = float(match.group(7)) if match.group(7) else None
                val_loss = float(match.group(8)) if match.group(8) else None

                data.append({
                    'epoch': epoch,
                    'step': step,
                    'train_mask_loss': train_mask_loss,
                    'train_topo_loss': train_topo_loss,
                    # 'train_contrastive_loss': train_contrastive_loss,
                    'train_loss': train_loss,
                    'val_mask_loss': val_mask_loss,
                    'val_topo_loss': val_topo_loss,
                    'val_loss': val_loss,
                })

    df = pd.DataFrame(data)

    # 将每个 epoch 的 step 累加为全局 step
    global_step = 0
    last_epoch = -1
    step_offset = 0
    global_steps = []

    for i, row in df.iterrows():
        if row['epoch'] != last_epoch:
            if last_epoch >= 0:
                step_offset = global_step  # 记录前一轮的累加
            last_epoch = row['epoch']
        global_step = step_offset + row['step']
        global_steps.append(global_step)

    df['global_step'] = global_steps
    return df


def plot_loss(df, loss_key):
    fig = px.line(
        df,
        x='global_step',
        y=loss_key,
        color='epoch',
        hover_data=['epoch', 'step', 'global_step', loss_key],
        title=f"{loss_key} over training (global step)"
    )
    fig.update_layout(hovermode='x unified')
    fig.show()


if __name__ == "__main__":
    # log_path = "sam_road_contra_lambda001_more_epoch.txt"  # 替换为你的日志路径
    log_path = "sam_road.txt"
    df = parse_and_accumulate_log(log_path)

    # 绘制训练loss曲线（按global step累加）
    plot_loss(df, 'train_loss')
    plot_loss(df, 'train_mask_loss')
    plot_loss(df, 'train_topo_loss')
    # plot_loss(df, 'train_contrastive_loss')

    # 验证loss（每个epoch一次）可以用 epoch 作为x轴
    val_df = df.dropna(subset=['val_loss']).drop_duplicates('epoch', keep='last')
    plot_loss(val_df, 'val_loss')
    plot_loss(val_df, 'val_mask_loss')
    plot_loss(val_df, 'val_topo_loss')
