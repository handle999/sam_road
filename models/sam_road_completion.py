"""
SAM-Road Completion v2 Model
=============================
路网补全模型: 输入遥感影像(+可选轨迹热力图) + 已知路网 → 补全后的完整路网

核心设计 (方案B):
  - 路径A: traj_heatmap 作为 SAM Encoder 的第4通道输入 (浅层注入, 无traj时全零退化)
  - 路径B: 已知路网边通过 RoadGraphGNN 编码拓扑先验
  - road_feature_map: 2通道 (已知路mask + 已知节点位置), CNN编码后与视觉特征融合
  - 训练时: 所有数据集统一用随机采样GT作为已知路网
  - 推理时: 通过最近邻映射将已知路网边关联到NMS关键点索引

退化逻辑:
  - 无traj + 无已知路网 → 等价于原始 SAM-Road (extraction)
  - 无traj + 有已知路网 → Completion无traj模式
  - 有traj + 有已知路网 → Completion有traj模式 (效果最好)
"""

import torch
import torch.nn.functional as F
from torch import nn
import math
from functools import partial
from torchmetrics.classification import BinaryJaccardIndex, F1Score, BinaryPrecisionRecallCurve
import lightning.pytorch as pl
from sam.segment_anything.modeling.image_encoder import ImageEncoderViT
from sam.segment_anything.modeling.common import LayerNorm2d
import pprint
import torchvision


class BilinearSampler(nn.Module):
    """与原版完全相同的双线性采样器"""

    def __init__(self, config):
        super(BilinearSampler, self).__init__()
        self.config = config

    def forward(self, feature_maps, sample_points):
        B, D, H, W = feature_maps.shape
        _, N_points, _ = sample_points.shape
        sample_points = (sample_points / self.config.PATCH_SIZE) * 2.0 - 1.0
        sample_points = sample_points.unsqueeze(2)
        sampled_features = F.grid_sample(feature_maps, sample_points, mode='bilinear', align_corners=False)
        sampled_features = sampled_features.squeeze(dim=-1).permute(0, 2, 1)
        return sampled_features


class TopoNetCompletion(nn.Module):
    """
    路网补全版 TopoNet

    相比原版 TopoNet 的修改:
      - pair_proj 输入维度增加: 2*D + 2 → 2*D + 2 + 2*graph_dim
      - 新增 graph_proj: 将 GNN 输出的图拓扑嵌入拼接到候选边特征中
      - graph_embeddings=None 时退化为原版 TopoNet (方便消融)
    """

    def __init__(self, config, feature_dim, graph_dim=32):
        super(TopoNetCompletion, self).__init__()
        self.config = config
        self.hidden_dim = 128
        self.heads = 4
        self.num_attn_layers = 3
        self.graph_dim = graph_dim

        self.feature_proj = nn.Linear(feature_dim, self.hidden_dim)
        # 原版: 2 * hidden_dim + 2
        # 补全版: 2 * hidden_dim + 2 + 2 * graph_dim (src_graph_embed + tgt_graph_embed)
        self.pair_proj = nn.Linear(2 * self.hidden_dim + 2 + 2 * self.graph_dim, self.hidden_dim)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.hidden_dim,
            nhead=self.heads,
            dim_feedforward=self.hidden_dim,
            dropout=0.1,
            activation='relu',
            batch_first=True
        )

        if self.config.TOPONET_VERSION != 'no_transformer':
            self.transformer_encoder = nn.TransformerEncoder(
                encoder_layer, num_layers=self.num_attn_layers
            )
        self.output_proj = nn.Linear(self.hidden_dim, 1)

    def forward(self, points, point_features, pairs, pairs_valid, graph_embeddings=None):
        """
        Args:
            points: [B, N_points, 2] 图节点坐标
            point_features: [B, N_points, D] 节点视觉特征 (来自融合后的特征图)
            pairs: [B, N_samples, N_pairs, 2] 候选边索引
            pairs_valid: [B, N_samples, N_pairs] 有效边掩码
            graph_embeddings: [B, N_points, graph_dim] GNN 编码的图拓扑嵌入
                              None 时退化为原版 TopoNet
        Returns:
            logits: [B, N_samples, N_pairs, 1]
            scores: [B, N_samples, N_pairs, 1]
        """
        point_features = F.relu(self.feature_proj(point_features))

        batch_size, n_samples, n_pairs, _ = pairs.shape
        pairs = pairs.view(batch_size, -1, 2)

        batch_indices = torch.arange(batch_size, device=points.device).view(-1, 1).expand(-1, n_samples * n_pairs)
        src_features = point_features[batch_indices, pairs[:, :, 0]]
        tgt_features = point_features[batch_indices, pairs[:, :, 1]]
        src_points = points[batch_indices, pairs[:, :, 0]]
        tgt_points = points[batch_indices, pairs[:, :, 1]]
        offset = tgt_points - src_points

        # 融入图拓扑嵌入
        if graph_embeddings is not None:
            src_graph = graph_embeddings[batch_indices, pairs[:, :, 0]]  # [B, S*P, graph_dim]
            tgt_graph = graph_embeddings[batch_indices, pairs[:, :, 1]]  # [B, S*P, graph_dim]
            pair_features = torch.concat([src_features, tgt_features, offset, src_graph, tgt_graph], dim=2)
        else:
            # 退化: 补零, 等价于原版 TopoNet
            graph_pad = torch.zeros(
                batch_size, n_samples * n_pairs, 2 * self.graph_dim,
                device=points.device, dtype=point_features.dtype
            )
            pair_features = torch.concat([src_features, tgt_features, offset, graph_pad], dim=2)

        pair_features = F.relu(self.pair_proj(pair_features))

        # attn applies within each local graph sample
        pair_features = pair_features.view(batch_size * n_samples, n_pairs, -1)
        pairs_valid = pairs_valid.view(batch_size * n_samples, n_pairs)

        all_invalid_pair_mask = torch.eq(torch.sum(pairs_valid, dim=-1), 0).unsqueeze(-1)
        pairs_valid = torch.logical_or(pairs_valid, all_invalid_pair_mask)
        padding_mask = ~pairs_valid

        if self.config.TOPONET_VERSION != 'no_transformer':
            pair_features = self.transformer_encoder(pair_features, src_key_padding_mask=padding_mask)

        _, n_pairs, _ = pair_features.shape
        pair_features = pair_features.view(batch_size, n_samples, n_pairs, -1)

        logits = self.output_proj(pair_features)
        scores = torch.sigmoid(logits)

        return logits, scores


class RoadGraphEncoder(nn.Module):
    """
    已知路网几何特征图编码器 (v2: 2通道版)

    将已知路网渲染的2通道特征图 (mask + 节点位置)
    编码为与 image_embeddings 维度对齐的特征图 [B, 256, 16, 16]

    通道说明 (v2精简版):
      - ch0: 已知道路 mask (哪里有路) — CNN无法从RGB 100%确定, 是强先验
      - ch1: 已知节点位置 (确定的路网节点) — 区分已知/未知节点
    """

    def __init__(self, output_dim=256):
        super(RoadGraphEncoder, self).__init__()
        self.encoder = nn.Sequential(
            # 输入: [B, 2, H, W] (H=W=PATCH_SIZE, 如256)
            nn.Conv2d(2, 32, 3, padding=1), nn.BatchNorm2d(32), nn.GELU(),
            nn.Conv2d(32, 64, 3, stride=4, padding=1), nn.BatchNorm2d(64), nn.GELU(),
            # → [B, 64, H/4, W/4]
            nn.Conv2d(64, 128, 3, stride=4, padding=1), nn.BatchNorm2d(128), nn.GELU(),
            # → [B, 128, H/16, W/16]
            nn.Conv2d(128, output_dim, 3, padding=1),
            # → [B, 256, H/16, W/16]
        )

    def forward(self, road_feature_map):
        """
        Args:
            road_feature_map: [B, 2, H, W] 渲染的已知路网特征图
        Returns:
            road_embeddings: [B, 256, H/16, W/16]
        """
        return self.encoder(road_feature_map)


class RoadGraphGNN(nn.Module):
    """
    已知路网拓扑结构 GNN 编码器

    使用 MultiheadAttention 在已知路网的边上做消息传递,
    为每个节点生成图拓扑嵌入。

    与几何特征图编码器互补:
      - 几何特征图: 解决节点对齐问题 (连续采样, 不怕偏移)
      - GNN: 理解拓扑结构 (哪些点之间已有连接, 补全不应违背已有拓扑)

    不对"需要补全"做任何假设, 只是客观编码"谁连谁"。
    """

    def __init__(self, node_dim=256, graph_dim=32, num_heads=4, num_layers=2):
        super(RoadGraphGNN, self).__init__()
        self.graph_dim = graph_dim

        # 节点视觉特征 → 图特征空间
        self.node_proj = nn.Linear(node_dim, graph_dim)

        # 多层图注意力 (用 MultiheadAttention 模拟 GAT)
        self.gat_layers = nn.ModuleList()
        self.norms = nn.ModuleList()
        for _ in range(num_layers):
            self.gat_layers.append(
                nn.MultiheadAttention(embed_dim=graph_dim, num_heads=num_heads, batch_first=True)
            )
            self.norms.append(nn.LayerNorm(graph_dim))

        # 坐标编码 (让 GNN 感知空间关系)
        self.coord_proj = nn.Linear(2, graph_dim)

    def forward(self, node_visual_features, node_coords, edge_index, node_mask=None):
        """
        Args:
            node_visual_features: [B, N, D] 从融合特征图采样的节点视觉特征
            node_coords: [B, N, 2] 节点坐标
            edge_index: [B, 2, E] 边的 (src, tgt) 索引 (已知路网的边)
            node_mask: [B, N] 有效节点掩码 (padding 的节点为 False)
        Returns:
            graph_embeddings: [B, N, graph_dim]
        """
        B, N, D = node_visual_features.shape

        # 投影到图特征空间
        x = self.node_proj(node_visual_features)  # [B, N, graph_dim]

        # 加入坐标信息
        coord_feat = self.coord_proj(node_coords)  # [B, N, graph_dim]
        x = x + coord_feat

        # 构造邻接注意力掩码
        # adj_mask: [B, N, N], True=可以注意, False=屏蔽
        if edge_index is not None and edge_index.shape[2] > 0:
            adj_mask = self._build_adj_mask(edge_index, B, N, node_coords.device)
            # 转换为 attn_mask 格式: float, 0=可以注意, -inf=屏蔽
            # PyTorch MHA 的 attn_mask 3D 格式: (B*num_heads, L, S)
            attn_mask = torch.zeros(B, N, N, device=node_coords.device, dtype=x.dtype)
            attn_mask = attn_mask.masked_fill(~adj_mask, float('-inf'))
            # 扩展到 num_heads: [B, N, N] -> [B*num_heads, N, N]
            attn_mask = attn_mask.repeat_interleave(self.gat_layers[0].num_heads, dim=0)
        else:
            # 没有已知路网边, 全连接注意力
            attn_mask = None

        # 多层图注意力 + 残差
        for gat, norm in zip(self.gat_layers, self.norms):
            residual = x
            # MultiheadAttention 需要 (L, B, E) 或 (B, L, E) with batch_first=True
            x_attn, _ = gat(x, x, x, attn_mask=attn_mask)
            x = norm(residual + x_attn)

        return x

    def _build_adj_mask(self, edge_index, B, N, device):
        """从 edge_index 构造邻接掩码矩阵"""
        mask = torch.zeros(B, N, N, device=device, dtype=torch.bool)
        # 加入自环 (每个节点可以注意自己)
        diag = torch.arange(N, device=device)
        mask[:, diag, diag] = True

        src = edge_index[:, 0, :]  # [B, E]
        tgt = edge_index[:, 1, :]  # [B, E]

        # 边索引可能超出范围, 需要 clamp
        valid_edge = (src >= 0) & (src < N) & (tgt >= 0) & (tgt < N)

        for b in range(B):
            ve = valid_edge[b]
            if ve.any():
                s = src[b, ve].long()
                t = tgt[b, ve].long()
                mask[b, s, t] = True
                mask[b, t, s] = True  # 无向图

        return mask


class _LoRA_qkv(nn.Module):
    """与原版完全相同的 LoRA 实现"""

    def __init__(self, qkv, linear_a_q, linear_b_q, linear_a_v, linear_b_v):
        super().__init__()
        self.weight = qkv.weight
        self.bias = qkv.bias
        self.linear_a_q = linear_a_q
        self.linear_b_q = linear_b_q
        self.linear_a_v = linear_a_v
        self.linear_b_v = linear_b_v
        self.dim = qkv.in_features

    def forward(self, x):
        qkv = F.linear(x, self.weight, self.bias)
        new_q = self.linear_b_q(self.linear_a_q(x))
        new_v = self.linear_b_v(self.linear_a_v(x))
        qkv[:, :, :, :self.dim] += new_q
        qkv[:, :, :, -self.dim:] += new_v
        return qkv


class SAMRoadCompletion(pl.LightningModule):
    """
    SAM-Road 路网补全模型 v2

    输入:
      - rgb [B, H, W, 3] + traj_heatmap [B, H, W, 1] → concat为 [B, H, W, 4]
        (无traj时 traj_heatmap=zeros, 等价于3ch输入, 但patch_embed第4通道零初始化保证退化)
      - 已知路网渲染特征图 [B, 2, H, W] (mask + 节点位置)
      - 已知路网边索引 [B, 2, E] (用于 GNN)
    输出:
      - 分割 mask (keypoint + road)
      - 候选边连接概率

    路径A: traj_heatmap → SAM第4通道 → 更好的image_embeddings
    路径B: known_edge_index → GNN → 更好的拓扑预测
    road_feature_map → CNN → 与image_embeddings融合 → fused_features
    """

    def __init__(self, config):
        super().__init__()
        self.config = config

        # ---- SAM 配置 ----
        assert config.SAM_VERSION in {'vit_b', 'vit_l', 'vit_h'}
        if config.SAM_VERSION == 'vit_b':
            encoder_embed_dim = 768
            encoder_depth = 12
            encoder_num_heads = 12
            encoder_global_attn_indexes = [2, 5, 8, 11]
        elif config.SAM_VERSION == 'vit_l':
            encoder_embed_dim = 1024
            encoder_depth = 24
            encoder_num_heads = 16
            encoder_global_attn_indexes = [5, 11, 17, 23]
        elif config.SAM_VERSION == 'vit_h':
            encoder_embed_dim = 1280
            encoder_depth = 32
            encoder_num_heads = 16
            encoder_global_attn_indexes = [7, 15, 23, 31]

        prompt_embed_dim = 256
        image_size = config.PATCH_SIZE
        self.image_size = image_size
        vit_patch_size = 16
        image_embedding_size = image_size // vit_patch_size
        encoder_output_dim = prompt_embed_dim

        # ---- 路径A: 4通道 pixel_mean/std ----
        # 第4通道(traj_heatmap)取值0~1, mean=0, std=1, 不影响原始分布
        self.register_buffer("pixel_mean", torch.Tensor([123.675, 116.28, 103.53, 0.0]).view(-1, 1, 1), False)
        self.register_buffer("pixel_std", torch.Tensor([58.395, 57.12, 57.375, 1.0]).view(-1, 1, 1), False)

        # ---- SAM Image Encoder ----
        if self.config.NO_SAM:
            raise NotImplementedError("NO_SAM mode not supported in completion model")
        else:
            self.image_encoder = ImageEncoderViT(
                depth=encoder_depth,
                embed_dim=encoder_embed_dim,
                img_size=image_size,
                mlp_ratio=4,
                norm_layer=partial(torch.nn.LayerNorm, eps=1e-6),
                num_heads=encoder_num_heads,
                patch_size=vit_patch_size,
                qkv_bias=True,
                use_rel_pos=True,
                global_attn_indexes=encoder_global_attn_indexes,
                window_size=14,
                out_chans=prompt_embed_dim
            )

        # ---- 路径A: 修改 patch_embed 为 4 通道 ----
        old_proj = self.image_encoder.patch_embed.proj
        out_ch, _, k_h, k_w = old_proj.weight.shape
        new_proj = nn.Conv2d(4, out_ch, kernel_size=(k_h, k_w), stride=(k_h, k_w), bias=True)
        self.image_encoder.patch_embed.proj = new_proj

        # ---- Map Decoder (不变, 与原版相同, 只用image_embeddings) ----
        if self.config.USE_SAM_DECODER:
            from sam.segment_anything.modeling.mask_decoder import MaskDecoder
            from sam.segment_anything.modeling.prompt_encoder import PromptEncoder
            from sam.segment_anything.modeling.transformer import TwoWayTransformer

            self.prompt_encoder = PromptEncoder(
                embed_dim=prompt_embed_dim,
                image_embedding_size=(image_embedding_size, image_embedding_size),
                input_image_size=(image_size, image_size),
                mask_in_chans=16,
            )
            for param in self.prompt_encoder.parameters():
                param.requires_grad = False
            self.mask_decoder = MaskDecoder(
                num_multimask_outputs=2,
                transformer=TwoWayTransformer(
                    depth=2, embedding_dim=prompt_embed_dim,
                    mlp_dim=2048, num_heads=8,
                ),
                transformer_dim=prompt_embed_dim,
                iou_head_depth=3, iou_head_hidden_dim=256,
            )
        else:
            activation = nn.GELU
            self.map_decoder = nn.Sequential(
                nn.ConvTranspose2d(encoder_output_dim, 128, kernel_size=2, stride=2),
                LayerNorm2d(128), activation(),
                nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2),
                activation(),
                nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2),
                activation(),
                nn.ConvTranspose2d(32, 2, kernel_size=2, stride=2),
            )

        # ---- 已知路网几何特征编码器 (2通道版) ----
        self.road_feat_encoder = RoadGraphEncoder(output_dim=encoder_output_dim)

        # ---- 视觉特征 + 路网特征融合 ----
        self.feature_fusion = nn.Sequential(
            nn.Conv2d(encoder_output_dim + encoder_output_dim, encoder_output_dim, 1),
            nn.GELU(),
            nn.Conv2d(encoder_output_dim, encoder_output_dim, 1),
        )

        # ---- 已知路网 GNN 编码器 ----
        graph_dim = getattr(config, 'GRAPH_DIM', 32)
        self.graph_dim = graph_dim
        self.road_graph_gnn = RoadGraphGNN(
            node_dim=encoder_output_dim, graph_dim=graph_dim,
            num_heads=4, num_layers=2
        )

        # ---- TopoNet (改进版) ----
        self.bilinear_sampler = BilinearSampler(config)
        self.topo_net = TopoNetCompletion(config, encoder_output_dim, graph_dim=graph_dim)

        # ---- LoRA ----
        if config.ENCODER_LORA:
            r = self.config.LORA_RANK
            assert r > 0
            self.lora_layer_selection = list(range(len(self.image_encoder.blocks)))
            self.w_As = []
            self.w_Bs = []
            for param in self.image_encoder.parameters():
                param.requires_grad = False

            # 路径A: 强制解冻 patch_embed (第4通道必须训练)
            self.image_encoder.patch_embed.proj.weight.requires_grad = True
            if self.image_encoder.patch_embed.proj.bias is not None:
                self.image_encoder.patch_embed.proj.bias.requires_grad = True

            for t_layer_i, blk in enumerate(self.image_encoder.blocks):
                if t_layer_i not in self.lora_layer_selection:
                    continue
                w_qkv_linear = blk.attn.qkv
                dim = w_qkv_linear.in_features
                w_a_linear_q = nn.Linear(dim, r, bias=False)
                w_b_linear_q = nn.Linear(r, dim, bias=False)
                w_a_linear_v = nn.Linear(dim, r, bias=False)
                w_b_linear_v = nn.Linear(r, dim, bias=False)
                self.w_As.append(w_a_linear_q)
                self.w_Bs.append(w_b_linear_q)
                self.w_As.append(w_a_linear_v)
                self.w_Bs.append(w_b_linear_v)
                blk.attn.qkv = _LoRA_qkv(
                    w_qkv_linear, w_a_linear_q, w_b_linear_q, w_a_linear_v, w_b_linear_v
                )
            for w_A in self.w_As:
                nn.init.kaiming_uniform_(w_A.weight, a=math.sqrt(5))
            for w_B in self.w_Bs:
                nn.init.zeros_(w_B.weight)

        # ---- Losses ----
        if self.config.FOCAL_LOSS:
            self.mask_criterion = partial(torchvision.ops.sigmoid_focal_loss, reduction='mean')
        else:
            self.mask_criterion = torch.nn.BCEWithLogitsLoss()
        self.topo_criterion = torch.nn.BCEWithLogitsLoss(reduction='none')

        # ---- Metrics ----
        self.keypoint_iou = BinaryJaccardIndex(threshold=0.5)
        self.road_iou = BinaryJaccardIndex(threshold=0.5)
        self.topo_f1 = F1Score(task='binary', threshold=0.5, ignore_index=-1)
        self.keypoint_pr_curve = BinaryPrecisionRecallCurve(ignore_index=-1)
        self.road_pr_curve = BinaryPrecisionRecallCurve(ignore_index=-1)
        self.topo_pr_curve = BinaryPrecisionRecallCurve(ignore_index=-1)

        # ---- Load SAM checkpoint ----
        if self.config.NO_SAM:
            return
        with open(config.SAM_CKPT_PATH, "rb") as f:
            ckpt_state_dict = torch.load(f, map_location='cpu')
            if image_size != 1024:
                ckpt_state_dict = self.resize_sam_pos_embed(
                    ckpt_state_dict, image_size, vit_patch_size, encoder_global_attn_indexes
                )

            matched_names = []
            mismatch_names = []
            state_dict_to_load = {}
            for k, v in self.named_parameters():
                if k in ckpt_state_dict and v.shape == ckpt_state_dict[k].shape:
                    matched_names.append(k)
                    state_dict_to_load[k] = ckpt_state_dict[k]
                elif k == 'image_encoder.patch_embed.proj.weight' and k in ckpt_state_dict:
                    # 路径A: 3通道权重拷贝给前3通道, 第4通道零初始化
                    print(f"[{k}] Adapting SAM weights from 3 channels to 4 channels.")
                    old_weight = ckpt_state_dict[k]
                    new_weight = torch.zeros_like(v)
                    new_weight[:, :3, :, :] = old_weight
                    # 第4通道保持全零初始化
                    matched_names.append(k)
                    state_dict_to_load[k] = new_weight
                else:
                    mismatch_names.append(k)
            print("###### Matched params ######")
            pprint.pprint(matched_names)
            print("###### Mismatched params (completion-specific) ######")
            pprint.pprint(mismatch_names)

            self.matched_param_names = set(matched_names)
            self.load_state_dict(state_dict_to_load, strict=False)

    def resize_sam_pos_embed(self, state_dict, image_size, vit_patch_size, encoder_global_attn_indexes):
        """与原版完全相同"""
        new_state_dict = {k: v for k, v in state_dict.items()}
        pos_embed = new_state_dict['image_encoder.pos_embed']
        token_size = int(image_size // vit_patch_size)
        if pos_embed.shape[1] != token_size:
            pos_embed = pos_embed.permute(0, 3, 1, 2)
            pos_embed = F.interpolate(pos_embed, (token_size, token_size), mode='bilinear', align_corners=False)
            pos_embed = pos_embed.permute(0, 2, 3, 1)
            new_state_dict['image_encoder.pos_embed'] = pos_embed
            rel_pos_keys = [k for k in state_dict.keys() if 'rel_pos' in k]
            global_rel_pos_keys = [k for k in rel_pos_keys if any([str(i) in k for i in encoder_global_attn_indexes])]
            for k in global_rel_pos_keys:
                rel_pos_params = new_state_dict[k]
                h, w = rel_pos_params.shape
                rel_pos_params = rel_pos_params.unsqueeze(0).unsqueeze(0)
                rel_pos_params = F.interpolate(rel_pos_params, (token_size * 2 - 1, w), mode='bilinear', align_corners=False)
                new_state_dict[k] = rel_pos_params[0, 0, ...]
        return new_state_dict

    def forward(self, rgb, traj_heatmap, road_feature_map, graph_points, pairs, valid, known_edge_index=None):
        """
        Args:
            rgb: [B, H, W, 3] 遥感影像
            traj_heatmap: [B, H, W, 1] 轨迹热力图 (无traj时全零)
            road_feature_map: [B, 2, H, W] 已知路网渲染特征图 (mask + 节点位置)
            graph_points: [B, N_points, 2] 图节点坐标
            pairs: [B, N_samples, N_pairs, 2] 候选边索引
            valid: [B, N_samples, N_pairs] 有效边掩码
            known_edge_index: [B, 2, E] 已知路网边索引 (用于 GNN)
        Returns:
            mask_logits: [B, H, W, 2]
            mask_scores: [B, H, W, 2]
            topo_logits: [B, N_samples, N_pairs, 1]
            topo_scores: [B, N_samples, N_pairs, 1]
        """
        # ---- 路径A: 4通道输入 → SAM Encoder ----
        x = torch.cat([rgb, traj_heatmap], dim=3)  # [B, H, W, 4]
        x = x.permute(0, 3, 1, 2)  # [B, 4, H, W]
        x = (x - self.pixel_mean) / self.pixel_std
        image_embeddings = self.image_encoder(x)  # [B, 256, h, w]

        # ---- 已知路网几何特征编码 ----
        road_embeddings = self.road_feat_encoder(road_feature_map)  # [B, 256, h, w]

        # ---- 融合视觉 + 路网特征 ----
        fused_features = self.feature_fusion(
            torch.cat([image_embeddings, road_embeddings], dim=1)
        )  # [B, 256, h, w]

        # ---- 分割头 (只用image_embeddings, 不混入路网信息, 避免过拟合) ----
        if self.config.USE_SAM_DECODER:
            sparse_embeddings, dense_embeddings = self.prompt_encoder(
                points=None, boxes=None, masks=None
            )
            low_res_logits, _ = self.mask_decoder(
                image_embeddings=image_embeddings,
                image_pe=self.prompt_encoder.get_dense_pe(),
                sparse_prompt_embeddings=sparse_embeddings,
                dense_prompt_embeddings=dense_embeddings,
                multimask_output=True
            )
            mask_logits = F.interpolate(
                low_res_logits,
                (self.image_encoder.img_size, self.image_encoder.img_size),
                mode="bilinear", align_corners=False,
            )
            mask_scores = torch.sigmoid(mask_logits)
        else:
            mask_logits = self.map_decoder(image_embeddings)
            mask_scores = torch.sigmoid(mask_logits)

        # ---- Stage 2: TopoNet (改进版) ----
        # 从融合特征图采样节点特征 (包含视觉 + 路网几何信息)
        point_features = self.bilinear_sampler(fused_features, graph_points)  # [B, N, 256]

        # 路径B: GNN 编码已知路网拓扑
        if known_edge_index is not None and known_edge_index.shape[2] > 0:
            graph_embeddings = self.road_graph_gnn(
                point_features, graph_points, known_edge_index
            )  # [B, N, graph_dim]
        else:
            graph_embeddings = None

        topo_logits, topo_scores = self.topo_net(
            graph_points, point_features, pairs, valid, graph_embeddings
        )

        mask_logits = mask_logits.permute(0, 2, 3, 1)
        mask_scores = mask_scores.permute(0, 2, 3, 1)
        return mask_logits, mask_scores, topo_logits, topo_scores

    def infer_masks_and_img_features(self, rgb, traj_heatmap, road_feature_map):
        """
        推理阶段: 获取分割 mask 和融合特征图

        Args:
            rgb: [B, H, W, 3]
            traj_heatmap: [B, H, W, 1]
            road_feature_map: [B, 2, H, W]
        Returns:
            mask_scores: [B, H, W, 2]
            fused_features: [B, 256, h, w]
        """
        x = torch.cat([rgb, traj_heatmap], dim=3)  # [B, H, W, 4]
        x = x.permute(0, 3, 1, 2)  # [B, 4, H, W]
        x = (x - self.pixel_mean) / self.pixel_std
        image_embeddings = self.image_encoder(x)

        road_embeddings = self.road_feat_encoder(road_feature_map)
        fused_features = self.feature_fusion(
            torch.cat([image_embeddings, road_embeddings], dim=1)
        )

        if self.config.USE_SAM_DECODER:
            sparse_embeddings, dense_embeddings = self.prompt_encoder(
                points=None, boxes=None, masks=None
            )
            low_res_logits, _ = self.mask_decoder(
                image_embeddings=image_embeddings,
                image_pe=self.prompt_encoder.get_dense_pe(),
                sparse_prompt_embeddings=sparse_embeddings,
                dense_prompt_embeddings=dense_embeddings,
                multimask_output=True
            )
            mask_logits = F.interpolate(
                low_res_logits,
                (self.image_encoder.img_size, self.image_encoder.img_size),
                mode="bilinear", align_corners=False,
            )
            mask_scores = torch.sigmoid(mask_logits)
        else:
            mask_logits = self.map_decoder(image_embeddings)
            mask_scores = torch.sigmoid(mask_logits)

        mask_scores = mask_scores.permute(0, 2, 3, 1)
        return mask_scores, fused_features

    def infer_toponet(self, fused_features, graph_points, pairs, valid, known_edge_index=None):
        """
        推理阶段: TopoNet 预测

        Args:
            fused_features: [B, 256, h, w] 融合特征图
            graph_points: [B, N_points, 2]
            pairs: [B, N_samples, N_pairs, 2]
            valid: [B, N_samples, N_pairs]
            known_edge_index: [B, 2, E]
        Returns:
            topo_scores: [B, N_samples, N_pairs, 1]
        """
        point_features = self.bilinear_sampler(fused_features, graph_points)

        if known_edge_index is not None and known_edge_index.shape[2] > 0:
            graph_embeddings = self.road_graph_gnn(
                point_features, graph_points, known_edge_index
            )
        else:
            graph_embeddings = None

        _, topo_scores = self.topo_net(graph_points, point_features, pairs, valid, graph_embeddings)
        return topo_scores

    def training_step(self, batch, batch_idx):
        rgb = batch['rgb']  # [B, H, W, 3]
        traj_heatmap = batch.get('traj_heatmap', torch.zeros(rgb.shape[0], rgb.shape[1], rgb.shape[2], 1,
                                                               dtype=rgb.dtype, device=rgb.device))
        keypoint_mask, road_mask = batch['keypoint_mask'], batch['road_mask']
        graph_points, pairs, valid = batch['graph_points'], batch['pairs'], batch['valid']
        road_feature_map = batch['road_feature_map']
        known_edge_index = batch.get('known_edge_index', None)

        mask_logits, mask_scores, topo_logits, topo_scores = self(
            rgb, traj_heatmap, road_feature_map, graph_points, pairs, valid, known_edge_index
        )

        gt_masks = torch.stack([keypoint_mask, road_mask], dim=3)
        mask_loss = self.mask_criterion(mask_logits, gt_masks)

        topo_gt = batch['connected'].to(torch.int32)
        topo_loss_mask = valid.to(torch.float32)
        topo_loss = self.topo_criterion(topo_logits, topo_gt.unsqueeze(-1).to(torch.float32))
        topo_loss *= topo_loss_mask.unsqueeze(-1)
        topo_loss = topo_loss.sum() / topo_loss_mask.sum()

        loss = mask_loss + topo_loss
        self.log('train_mask_loss', mask_loss, on_step=True, on_epoch=False, prog_bar=True)
        self.log('train_topo_loss', topo_loss, on_step=True, on_epoch=False, prog_bar=True)
        self.log('train_loss', loss, on_step=True, on_epoch=False, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        rgb = batch['rgb']
        traj_heatmap = batch.get('traj_heatmap', torch.zeros(rgb.shape[0], rgb.shape[1], rgb.shape[2], 1,
                                                               dtype=rgb.dtype, device=rgb.device))
        keypoint_mask, road_mask = batch['keypoint_mask'], batch['road_mask']
        graph_points, pairs, valid = batch['graph_points'], batch['pairs'], batch['valid']
        road_feature_map = batch['road_feature_map']
        known_edge_index = batch.get('known_edge_index', None)

        mask_logits, mask_scores, topo_logits, topo_scores = self(
            rgb, traj_heatmap, road_feature_map, graph_points, pairs, valid, known_edge_index
        )

        gt_masks = torch.stack([keypoint_mask, road_mask], dim=3)
        mask_loss = self.mask_criterion(mask_logits, gt_masks)

        topo_gt = batch['connected'].to(torch.int32)
        topo_loss_mask = valid.to(torch.float32)
        topo_loss = self.topo_criterion(topo_logits, topo_gt.unsqueeze(-1).to(torch.float32))
        topo_loss *= topo_loss_mask.unsqueeze(-1)
        topo_loss = topo_loss.sum() / topo_loss_mask.sum()

        loss = mask_loss + topo_loss
        self.log('val_mask_loss', mask_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log('val_topo_loss', topo_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True)

        self.keypoint_iou.update(mask_scores[..., 0], keypoint_mask)
        self.road_iou.update(mask_scores[..., 1], road_mask)

        valid_int = valid.to(torch.int32)
        topo_gt_masked = (1 - valid_int) * -1 + valid_int * topo_gt
        self.topo_f1.update(topo_scores, topo_gt_masked.unsqueeze(-1))

    def on_validation_epoch_end(self):
        keypoint_iou = self.keypoint_iou.compute()
        road_iou = self.road_iou.compute()
        topo_f1 = self.topo_f1.compute()
        self.log("keypoint_iou", keypoint_iou)
        self.log("road_iou", road_iou)
        self.log("topo_f1", topo_f1)
        self.keypoint_iou.reset()
        self.road_iou.reset()
        self.topo_f1.reset()

    def test_step(self, batch, batch_idx):
        rgb = batch['rgb']
        traj_heatmap = batch.get('traj_heatmap', torch.zeros(rgb.shape[0], rgb.shape[1], rgb.shape[2], 1,
                                                               dtype=rgb.dtype, device=rgb.device))
        keypoint_mask, road_mask = batch['keypoint_mask'], batch['road_mask']
        graph_points, pairs, valid = batch['graph_points'], batch['pairs'], batch['valid']
        road_feature_map = batch['road_feature_map']
        known_edge_index = batch.get('known_edge_index', None)

        mask_logits, mask_scores, topo_logits, topo_scores = self(
            rgb, traj_heatmap, road_feature_map, graph_points, pairs, valid, known_edge_index
        )

        topo_gt = batch['connected'].to(torch.int32)
        valid_int = valid.to(torch.int32)

        self.keypoint_pr_curve.update(mask_scores[..., 0], keypoint_mask.to(torch.int32))
        self.road_pr_curve.update(mask_scores[..., 1], road_mask.to(torch.int32))

        topo_gt_masked = (1 - valid_int) * -1 + valid_int * topo_gt
        self.topo_pr_curve.update(topo_scores, topo_gt_masked.unsqueeze(-1).to(torch.int32))

    def on_test_end(self):
        def find_best_threshold(pr_curve_metric, category):
            print(f'======= {category} ======')
            precision, recall, thresholds = pr_curve_metric.compute()
            f1_scores = 2 * (precision * recall) / (precision + recall)
            best_threshold_index = torch.argmax(f1_scores)
            best_threshold = thresholds[best_threshold_index]
            best_precision = precision[best_threshold_index]
            best_recall = recall[best_threshold_index]
            best_f1 = f1_scores[best_threshold_index]
            print(f'Best threshold {best_threshold}, P={best_precision} R={best_recall} F1={best_f1}')

        print('======= Finding best thresholds ======')
        find_best_threshold(self.keypoint_pr_curve, 'keypoint')
        find_best_threshold(self.road_pr_curve, 'road')
        find_best_threshold(self.topo_pr_curve, 'topo')

    def configure_optimizers(self):
        param_dicts = []

        if not self.config.FREEZE_ENCODER and not self.config.ENCODER_LORA:
            encoder_params = {
                'params': [p for k, p in self.image_encoder.named_parameters()
                           if 'image_encoder.' + k in self.matched_param_names],
                'lr': self.config.BASE_LR * self.config.ENCODER_LR_FACTOR,
            }
            param_dicts.append(encoder_params)
        if self.config.ENCODER_LORA:
            encoder_params = {
                'params': [p for k, p in self.image_encoder.named_parameters()
                           if 'qkv.linear_' in k or 'patch_embed' in k],
                'lr': self.config.BASE_LR,
            }
            param_dicts.append(encoder_params)

        if self.config.USE_SAM_DECODER:
            decoder_params = [
                {'params': [p for k, p in self.mask_decoder.named_parameters()
                            if 'mask_decoder.' + k in self.matched_param_names],
                 'lr': self.config.BASE_LR * 0.1},
                {'params': [p for k, p in self.mask_decoder.named_parameters()
                            if 'mask_decoder.' + k not in self.matched_param_names],
                 'lr': self.config.BASE_LR}
            ]
        else:
            decoder_params = [{'params': list(self.map_decoder.parameters()), 'lr': self.config.BASE_LR}]
        param_dicts += decoder_params

        # 补全模型新增模块的参数
        completion_params = {
            'params': (
                list(self.road_feat_encoder.parameters()) +
                list(self.feature_fusion.parameters()) +
                list(self.road_graph_gnn.parameters()) +
                list(self.topo_net.parameters())
            ),
            'lr': self.config.BASE_LR
        }
        param_dicts.append(completion_params)

        for i, param_dict in enumerate(param_dicts):
            param_num = sum([int(p.numel()) for p in param_dict['params']])
            print(f'optim param dict {i} params num: {param_num}')

        optimizer = torch.optim.Adam(param_dicts, lr=self.config.BASE_LR)
        step_lr = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[9, ], gamma=0.1)
        return {'optimizer': optimizer, 'lr_scheduler': step_lr}
