"""可视化 spacenet 黑边 region: sat + gt.png + partial.png 并排"""
import cv2, numpy as np, os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

imgs = ['AOI_2_Vegas_505', 'AOI_2_Vegas_1173', 'AOI_4_Shanghai_1597']
base = 'datasets/spacenet/RGB_1.0_meter'

fig, axes = plt.subplots(len(imgs), 3, figsize=(12, 4*len(imgs)))
for r, img in enumerate(imgs):
    sat = cv2.imread(f'{base}/{img}__rgb.png')[:,:,::-1]
    gt = cv2.imread(f'{base}/{img}__gt.png', cv2.IMREAD_GRAYSCALE)
    pm = cv2.imread(f'{base}/{img}__gt_graph_partial.png', cv2.IMREAD_GRAYSCALE)
    black = (sat.max(axis=2) < 15).mean()
    axes[r,0].imshow(sat); axes[r,0].set_title(f'{img} sat (黑边{black:.0%})')
    axes[r,0].axis('off')
    axes[r,1].imshow(gt, cmap='gray'); axes[r,1].set_title('gt.png (rn)')
    axes[r,1].axis('off')
    axes[r,2].imshow(pm, cmap='gray'); axes[r,2].set_title('partial.png')
    axes[r,2].axis('off')
plt.tight_layout()
os.makedirs('docs/imgs', exist_ok=True)
plt.savefig('docs/imgs/spacenet_black_edge.png', dpi=100, bbox_inches='tight')
print('✓ docs/imgs/spacenet_black_edge.png')
