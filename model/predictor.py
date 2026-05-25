"""
Financial MoE Transformer v3 — Dual Embedding + Hierarchical Encoder.
"""
import torch
import torch.nn as nn

from .embedding import EmbeddingLayer
from .hierarchical import SimpleHierarchicalEncoder
from .moe import MoEPredictor


class AttentionPooling(nn.Module):
    def __init__(self, d_model=128, nhead=4):
        super().__init__()
        self.query = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)
        self.key_proj = nn.Linear(d_model, d_model)
        self.nhead = nhead
        self.d_head = d_model // nhead
        self.scale = self.d_head ** -0.5

    def forward(self, x, mask=None):
        B, L, D = x.shape
        Q = self.query.expand(B, -1, -1)
        K = self.key_proj(x)
        Q_h = Q.view(B, 1, self.nhead, self.d_head).transpose(1, 2)
        K_h = K.view(B, L, self.nhead, self.d_head).transpose(1, 2)
        attn = torch.matmul(Q_h, K_h.transpose(-2, -1)) * self.scale
        if mask is not None:
            attn = attn.masked_fill(~mask.unsqueeze(1).unsqueeze(2), -1e9)
        attn_w = torch.softmax(attn, dim=-1)
        V_h = x.view(B, L, self.nhead, self.d_head).transpose(1, 2)
        pooled = torch.matmul(attn_w, V_h)
        pooled = pooled.transpose(1, 2).contiguous().view(B, 1, D)
        return pooled.squeeze(1)


class FinancialMoETransformer(nn.Module):
    """v3: Dual embedding + hierarchical encoder + MoE."""

    def __init__(self, n_companies=5000, n_metrics=600,
                 d_model=128, nhead=4, num_layers=2, dim_feedforward=512,
                 dropout=0.2, attn_dropout=0.2, activation="gelu",
                 attn_type="cosformer", attn_chunk_size=2048,
                 num_experts=6, expert_hidden=64, top_k=1,
                 max_seq_len=8192, n_scales=4, scale_dim=32):
        super().__init__()
        self.d_model = d_model

        self.embedding = EmbeddingLayer(
            n_companies=n_companies, n_metrics=n_metrics, d_model=d_model)

        self.hierarchical = SimpleHierarchicalEncoder(
            d_model=d_model, nhead=nhead, num_layers=num_layers,
            dim_feedforward=dim_feedforward, dropout=dropout,
            attn_type=attn_type, attn_chunk_size=attn_chunk_size,
            n_scales=n_scales, scale_dim=scale_dim)

        self.pooling = AttentionPooling(d_model, nhead)

        self.moe = MoEPredictor(
            d_model=d_model, num_experts=num_experts,
            expert_hidden=expert_hidden, top_k=top_k)

        self._init_weights()

    def _init_weights(self):
        for name, param in self.named_parameters():
            if param.dim() >= 2 and "embedding" not in name and "query" not in name:
                nn.init.xavier_uniform_(param, gain=0.5)

    def forward(self, values, company_ids, metric_ids,
                day, month, dow, year_offset, time_since_update,
                mask=None, source_ids=None, time_bins=None):
        B, L = company_ids.shape
        device = company_ids.device
        source_ids = _ensure(source_ids, B, L, device)
        time_bins = _ensure(time_bins, B, L, device)
        mask = _ensure(mask, B, L, device, dtype=torch.bool, fill=True)

        x = self.embedding(values, company_ids, metric_ids,
                          day, month, dow, year_offset, time_since_update)
        x, hier_mask = self.hierarchical(x, mask, source_ids, time_bins)
        x_pooled = self.pooling(x, hier_mask)
        pred, lb_loss = self.moe(x_pooled)
        return pred, lb_loss

    def get_expert_weights(self, values, company_ids, metric_ids,
                           day, month, dow, year_offset, time_since_update,
                           mask=None, source_ids=None, time_bins=None):
        B, L = company_ids.shape
        device = company_ids.device
        source_ids = _ensure(source_ids, B, L, device)
        time_bins = _ensure(time_bins, B, L, device)
        mask = _ensure(mask, B, L, device, dtype=torch.bool, fill=True)
        x = self.embedding(values, company_ids, metric_ids,
                          day, month, dow, year_offset, time_since_update)
        x, hier_mask = self.hierarchical(x, mask, source_ids, time_bins)
        x_pooled = self.pooling(x, hier_mask)
        return self.moe.get_expert_weights(x_pooled)


def _ensure(tensor, B, L, device, dtype=torch.long, fill=False):
    """Ensure tensor is not None, create default if needed."""
    if tensor is not None:
        return tensor
    if fill:
        return torch.ones(B, L, dtype=torch.bool, device=device)
    return torch.zeros(B, L, dtype=dtype, device=device)
