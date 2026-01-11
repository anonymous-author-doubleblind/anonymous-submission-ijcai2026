# -*- coding: utf-8 -*-
# pycd/meta/selector.py
# 选择器 S：多层感知机，输出 K 个基模型的权重分布（softmax）
# 变更：
#   1) 默认隐藏层改为 200-200-200（可通过 --hidden 覆盖，对齐论文设置）
#   2) 支持 x-only / x+preds，两种输入；当拼接 base_preds 为特征时对该分支 detach
#   3) 保持与原有脚本的构造与 forward 接口一致
#
# 参考：MetaSelector 论文的设置：episode=10，selector 使用 200-200-200 MLP。参见论文第 5.3 节设置。  # noqa
# Luo et al., WWW'20, "We use a 200-200-200 MLP as the model selector." :contentReference[oaicite:3]{index=3}

from typing import Optional, Tuple, List
import torch
import torch.nn as nn
import torch.nn.functional as F

_ACTS = {
    "relu": nn.ReLU,
    "silu": nn.SiLU,
}

class SelectorMLP(nn.Module):
    """
    Selector S(x; φ): 给定 (student_id, exercise_id, q_vector[, base_preds])，输出对 K 个基模型的权重 λ。

    - x-only：只用 (student, exercise, q_vector) 构造特征；
    - x + M(x)：在 forward 传入 base_preds（形状 [B, K]），并设置 use_pred_as_feat=True，
      将 base_preds 作为额外特征拼接（该分支会 detach，避免与聚合主通路的梯度“叠加震荡”）。
    接口保持与原实现一致。:contentReference[oaicite:4]{index=4}
    """
    def __init__(
        self,
        n_students: int,
        n_exercises: int,
        n_concepts: int,
        n_bases: int,
        hidden: Tuple[int, ...] = (200, 200, 200),
        emb_dim: int = 16,
        qproj_dim: int = 16,
        act: str = "silu",          # "relu" / "silu"
        dropout: float = 0.2,
        use_pred_as_feat: bool = False,
    ):
        super().__init__()
        assert act in _ACTS, f"act must be one of {list(_ACTS.keys())}"
        self.use_pred_as_feat = use_pred_as_feat
        self.n_bases = n_bases

        # 离散特征嵌入
        self.student_emb = nn.Embedding(n_students, emb_dim)
        self.exercise_emb = nn.Embedding(n_exercises, emb_dim)

        # Q 向量压缩（支持多/稀疏概念的稠密化）
        self.q_proj = nn.Linear(n_concepts, qproj_dim, bias=True)

        # 输入维度：stu(emb_dim) + exer(emb_dim) + qproj(qproj_dim) [+ base_preds(K)]
        in_dim = emb_dim + emb_dim + qproj_dim + (n_bases if use_pred_as_feat else 0)

        # 构建 MLP：hidden 可为 1~N 层，默认 200-200-200（论文设置）:contentReference[oaicite:5]{index=5}
        Act = _ACTS[act]
        mlp_layers: List[nn.Module] = []
        prev = in_dim
        for h in hidden:
            mlp_layers += [nn.Linear(prev, h), Act(), nn.Dropout(dropout)]
            prev = h
        mlp_layers += [nn.Linear(prev, n_bases)]  # 输出未归一化权重
        self.mlp = nn.Sequential(*mlp_layers)

        # 参数初始化：用 xavier 均匀，偏置 0，更稳一些
        for m in self.mlp:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

        nn.init.xavier_uniform_(self.q_proj.weight)
        if self.q_proj.bias is not None:
            nn.init.zeros_(self.q_proj.bias)

        nn.init.normal_(self.student_emb.weight, std=0.02)
        nn.init.normal_(self.exercise_emb.weight, std=0.02)

    def forward(
        self,
        stu_id: torch.LongTensor,            # [B]
        exer_id: torch.LongTensor,           # [B]
        q_vector: torch.Tensor,              # [B, C]，已在外部对齐到稠密向量
        base_preds: Optional[torch.Tensor] = None  # [B, K]，当 use_pred_as_feat=True 时可传
    ) -> torch.Tensor:
        """
        返回：λ ∈ R^{B,K}，每行 softmax=1
        - 若 self.use_pred_as_feat=True 且提供 base_preds，则把其作为特征拼接（该分支 detach）
        - 最终 softmax 输出作为对 K 个基模型的加权系数
        """
        # 基础特征
        s = self.student_emb(stu_id)              # [B, E]
        e = self.exercise_emb(exer_id)            # [B, E]
        # 用 sigmoid 压一手，等价做个轻量非线性门控
        q = torch.sigmoid(self.q_proj(q_vector))  # [B, Qp]

        feats = [s, e, q]

        if self.use_pred_as_feat and base_preds is not None:
            # 仅作为特征的分支做 detach（聚合主通路不受影响）
            if base_preds.requires_grad:
                feats.append(base_preds.detach())
            else:
                feats.append(base_preds)

        x = torch.cat(feats, dim=1)               # [B, in_dim]
        logits = self.mlp(x)                      # [B, K]
        lam = F.softmax(logits, dim=-1)           # [B, K]，权重和为 1
        return lam
