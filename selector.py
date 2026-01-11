# -*- coding: utf-8 -*-
# pycd/meta/selector.py
# Selector S: an MLP that outputs a weight distribution (softmax) over K base models.
# Changes:
#   1) Default hidden layers set to 200-200-200 (can be overridden via --hidden; aligned with the paper)
#   2) Support two inputs: x-only / x+preds; when concatenating base_preds as features, detach that branch
#   3) Keep the constructor and forward interface consistent with the original scripts
#
# Reference: MetaSelector paper setting: episode=10, selector uses a 200-200-200 MLP. See Section 5.3.  # noqa
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
    Selector S(x; φ): given (student_id, exercise_id, q_vector[, base_preds]), output weights λ over K base models.

    - x-only: use only (student, exercise, q_vector) as features.
    - x + M(x): pass base_preds (shape [B, K]) in forward and set use_pred_as_feat=True,
      then concatenate base_preds as extra features (this branch is detached to avoid gradient coupling/instability).
    The interface is kept consistent with the original implementation. :contentReference[oaicite:4]{index=4}
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

        # Embeddings for discrete features
        self.student_emb = nn.Embedding(n_students, emb_dim)
        self.exercise_emb = nn.Embedding(n_exercises, emb_dim)

        # Q-vector projection (densify multi/sparse concepts)
        self.q_proj = nn.Linear(n_concepts, qproj_dim, bias=True)

        # Input dim: stu(emb_dim) + exer(emb_dim) + qproj(qproj_dim) [+ base_preds(K)]
        in_dim = emb_dim + emb_dim + qproj_dim + (n_bases if use_pred_as_feat else 0)

        # Build MLP: hidden can have 1~N layers; default 200-200-200 (paper setting) :contentReference[oaicite:5]{index=5}
        Act = _ACTS[act]
        mlp_layers: List[nn.Module] = []
        prev = in_dim
        for h in hidden:
            mlp_layers += [nn.Linear(prev, h), Act(), nn.Dropout(dropout)]
            prev = h
        mlp_layers += [nn.Linear(prev, n_bases)]  # output unnormalized weights
        self.mlp = nn.Sequential(*mlp_layers)

        # Init: Xavier uniform, zero bias (a bit more stable)
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
        q_vector: torch.Tensor,              # [B, C], already aligned to a dense vector externally
        base_preds: Optional[torch.Tensor] = None  # [B, K], used when use_pred_as_feat=True
    ) -> torch.Tensor:
        """
        Returns: λ ∈ R^{B,K}, softmax(row)=1
        - If self.use_pred_as_feat=True and base_preds is provided, concatenate it as features (detached branch)
        - The final softmax output is used as weights over K base models
        """
        # Base features
        s = self.student_emb(stu_id)              # [B, E]
        e = self.exercise_emb(exer_id)            # [B, E]
        # Apply a sigmoid for a light non-linear gating
        q = torch.sigmoid(self.q_proj(q_vector))  # [B, Qp]

        feats = [s, e, q]

        if self.use_pred_as_feat and base_preds is not None:
            # Detach the feature-only branch (the main aggregation path is unaffected)
            if base_preds.requires_grad:
                feats.append(base_preds.detach())
            else:
                feats.append(base_preds)

        x = torch.cat(feats, dim=1)               # [B, in_dim]
        logits = self.mlp(x)                      # [B, K]
        lam = F.softmax(logits, dim=-1)           # [B, K], weights sum to 1
        return lam
