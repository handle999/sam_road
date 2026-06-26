# Component 采样导致 completion 退化分析

> **症状**:partial 从"按边随机(破碎)"换成"按连通块保(component)"后,completion 效果大幅下降。
>
> | 数据集 | 0625(按边随机) | 0626(component) | 变化 |
> |--------|:---:|:---:|:---:|
> | spacenet | APLS 0.7000 | APLS 0.5293 | -0.171 |
> | didi_xian | APLS 0.5878 | APLS 0.4571 | -0.131 |

---

## 一、排查结论

### 1.1 唯一变量:训练侧采样策略

0625 vs 0626 的对比:
- config 完全一致(BATCH_SIZE/LR/KEEP_RATIO/MODALITY_DROPOUT 等)
- 输入形态一致(rn 都带,didi 都带 traj)
- ckpt 都是 epoch9
- 无 NaN(数值稳定)
- **唯一差异**:0625 训练侧 `_create_known_graph` = 按边随机;0626 = component

git commit:0625=`400f325`(按边随机),0626=`513dfc9`(component)。

### 1.2 训练正常,验证震荡

- **train_loss 正常**:0626 早期 train_loss(1.58→0.95)和 0625(1.61→0.96)几乎一样,甚至略低
- **val_loss 剧烈震荡**:0626 val_loss 在 0.20~1.28 间大幅跳动;0625 平滑收敛(0.18→0.215)
- train 正常但 val 震荡 → **模型学到的策略在 component 形态的 known graph 上表现不稳定**

### 1.3 不是 train/infer mismatch

两者都是 train+infer 同策略:
- 0625:训练(按边随机)+ 推理(按边随机破碎partial)= 一致 → 效果好
- 0626:训练(component)+ 推理(component连通partial)= 一致 → 效果差

所以**问题在 component 策略本身,不是 mismatch**。

---

## 二、根因分析:为什么 component 让效果变差

### 2.1 核心原因:监督信号分布不均

component 采样让 known graph **成块聚集**(整块留/整块删),而按边随机是**散布**。这导致:

- **某些 patch 全是已知边**(valid=False)→ 该 patch **无 topo 监督**(known 边不参与 loss)
- **某些 patch 全是未知边**(valid=True)→ 该 patch topo 监督密集
- 监督信号在 patch 间分布极不均匀

按边随机时,每个 patch 的已知/未知边比例较均匀(~50%),topo 监督信号稳定。component 让这个比例两极分化,训练信号变差。

### 2.2 completion 任务的特殊性

completion 的 topo loss **只在未知边(valid=True)上计算**。known edge 被标记 valid=False 不参与 loss。所以:
- 按边随机:每个 patch 都有约 50% 未知边 → 都有 topo 监督
- component:整块留的 patch 几乎全是已知边 → **无 topo 监督**;整块删的 patch 全未知 → 监督密集但缺先验上下文

这让模型见到的"已知+未知混合"训练样本变少,学不到"如何在已知路网基础上补全"的核心能力。

### 2.3 验证集也受影响

验证集 known graph 也是 component 采样(初始化时),val_loss 在 component 形态上评估,震荡反映模型对这种聚集形态泛化不稳。

---

## 三、这是一个重要的方法论发现

**"partial 不破碎"对真实场景合理,但对 completion 训练不一定更好。**

- **真实场景**:已知路网确实是不破碎的(整片完整)→ component 更真实
- **训练效率**:按边随机让每个 patch 都有混合的已知/未知边 → 监督信号充分,训练更稳

这两者矛盾。0625 的"破碎 partial"虽然在真实场景不自然,但**训练时每个 patch 都有已知+未知混合,监督信号均匀**,反而训得更好。

---

## 四、建议方案

### 方案A:回到按边随机(推荐,恢复效果)
- 训练侧:恢复按边随机(监督信号均匀)
- 推理侧:用按边随机生成的 partial(与训练一致)
- **代价**:partial 破碎(不自然),但效果恢复(0.70/0.59)
- 这其实是 0625 的状态,已验证有效

### 方案B:训练用按边随机,推理用 component(train/infer mismatch)
- 训练:按边随机(监督均匀)
- 推理:component partial(真实不破碎)
- **风险**:train/infer 分布不一致,可能效果打折。但模型见过各种破碎 partial,推理时 component 可能仍能处理
- 需实验验证

### 方案C:混合采样(折中)
- 训练时大部分按边随机 + 少部分 component,让模型两种形态都见过
- 推理用 component
- 实现复杂,需调比例

### 方案D:修改 loss 让 known edge 也参与(治本)
- 当前 known edge valid=False 不参与 topo loss,这是 component 监督不均的根源
- 可改为:known edge 也参与 loss(标签=1,因为已知存在),这样 component 也有监督
- 但改变了 completion 任务定义(模型不再"只补未知")

---

## 五、下一步建议

1. **立即**:用方案A 回退(恢复按边随机),确认效果回到 0.70/0.59
2. **再评估**:如果想要 component 的"真实不破碎",试方案B(训练随机/推理component)看能否兼顾
3. **长期**:方案D 是治本,但需重新设计 loss

> **关键认知**:"partial 不破碎"是真实场景需求,但当前 completion 的 loss 设计(known edge 不参与)让破碎 partial 反而训练更好。要兼得不破碎+好效果,需改 loss(方案D)或接受 train/infer mismatch(方案B)。
