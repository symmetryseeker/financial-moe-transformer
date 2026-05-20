"""
Sparse Mixture-of-Experts prediction head (v2 — larger experts).

Changes: 6 experts, hidden=64, Top-1 gating, load-balancing loss.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class Expert(nn.Module):
    def __init__(self, d_model=128, hidden=64, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden // 2, 1),
        )

    def forward(self, x):
        return self.net(x)


class MoEPredictor(nn.Module):
    """
    Top-1 sparse MoE with load-balancing auxiliary loss.

    Args:
        d_model: input dimension
        num_experts: number of expert networks
        expert_hidden: hidden dimension in each expert
        top_k: number of experts to activate per sample
    """

    def __init__(self, d_model=128, num_experts=6, expert_hidden=64,
                 top_k=1, expert_dropout=0.1):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k

        self.gate = nn.Linear(d_model, num_experts)
        self.experts = nn.ModuleList([
            Expert(d_model, expert_hidden, expert_dropout)
            for _ in range(num_experts)
        ])

        nn.init.normal_(self.gate.weight, mean=0.0, std=0.02)
        nn.init.zeros_(self.gate.bias)

    def forward(self, x):
        """
        Args:
            x: (B, d_model)

        Returns:
            pred: (B, 1)
            load_balance_loss: scalar
        """
        B = x.shape[0]

        gate_logits = self.gate(x)
        gate_probs = F.softmax(gate_logits, dim=-1)

        # Top-K sparse gating
        topk_probs, topk_idx = gate_probs.topk(self.top_k, dim=-1)
        topk_probs = topk_probs / topk_probs.sum(dim=-1, keepdim=True)

        # Route to experts: weighted sum of Top-K expert outputs
        pred = torch.zeros(B, 1, device=x.device)
        for e in range(self.num_experts):
            # Find which samples use expert e in their Top-K
            in_topk = (topk_idx == e)  # (B, K) bool
            if not in_topk.any():
                continue
            # Weight assigned to expert e per sample (0 if not in top-k)
            weights_e = torch.where(in_topk, topk_probs,
                                    torch.zeros_like(topk_probs)).sum(dim=-1)  # (B,)
            mask_e = weights_e > 0
            if mask_e.any():
                e_out = self.experts[e](x[mask_e])
                pred[mask_e] += e_out * weights_e[mask_e].unsqueeze(-1)

        # Load-balancing loss (Switch Transformer style)
        with torch.no_grad():
            routed = F.one_hot(topk_idx.squeeze(-1), self.num_experts).float()
            fraction = routed.mean(dim=0)
        mean_prob = gate_probs.mean(dim=0)
        load_balance_loss = self.num_experts * (fraction * mean_prob).sum()

        return pred, load_balance_loss

    def get_expert_weights(self, x):
        return F.softmax(self.gate(x), dim=-1)
