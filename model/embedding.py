"""
Multi-source token embedding v3 — dual-level (company + metric).

Solves the vocab explosion problem: instead of 92K unique variable embeddings
(sh_600519::close, sh_000001::close, ...), we split into:
  - company embedding (5000 companies × 32 dims)
  - metric embedding  (500 metrics × 96 dims)
  → total ~250K params instead of 11.8M.
"""

import torch
import torch.nn as nn
import math


class ValueProjection(nn.Module):
    def __init__(self, d_model=128):
        super().__init__()
        self.proj = nn.Linear(1, d_model)

    def forward(self, x):
        return self.proj(x)


class DualEmbedding(nn.Module):
    """
    Two-level embedding: company_id + metric_id → d_model.

    If a token has no company qualifier (e.g., macro data like "CPI"),
    company_id=0 is used (a shared "no-company" embedding).
    """

    def __init__(self, n_companies=5000, n_metrics=600,
                 company_dim=32, metric_dim=96, d_model=128):
        super().__init__()
        self.company_embed = nn.Embedding(n_companies, company_dim)
        self.metric_embed = nn.Embedding(n_metrics, metric_dim)
        self.d_model = d_model

        nn.init.normal_(self.company_embed.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.metric_embed.weight, mean=0.0, std=0.02)

    def forward(self, company_ids, metric_ids):
        """
        Args:
            company_ids: (B, L) int, 0 = no company qualifier
            metric_ids:  (B, L) int
        Returns:
            (B, L, d_model)
        """
        c = self.company_embed(company_ids)  # (B, L, company_dim)
        m = self.metric_embed(metric_ids)    # (B, L, metric_dim)
        return torch.cat([c, m], dim=-1)     # (B, L, d_model)


class CalendarEmbedding(nn.Module):
    def __init__(self, d_model=128, day_dim=8, month_dim=8, dow_dim=8, year_dim=8):
        super().__init__()
        self.day_embed = nn.Embedding(32, day_dim)
        self.month_embed = nn.Embedding(13, month_dim)
        self.dow_embed = nn.Embedding(8, dow_dim)
        self.year_embed = nn.Embedding(20, year_dim)
        total = day_dim + month_dim + dow_dim + year_dim
        self.proj = nn.Linear(total, d_model) if total != d_model else nn.Identity()

    def forward(self, day, month, dow, year_offset):
        d = self.day_embed(day.clamp(1, 31))
        m = self.month_embed(month.clamp(1, 12))
        w = self.dow_embed(dow.clamp(0, 7))
        y = self.year_embed(year_offset.clamp(0, 19))
        return self.proj(torch.cat([d, m, w, y], dim=-1))


class TimeBucketEmbedding(nn.Module):
    def __init__(self, bucket_edges=None, bucket_dim=8, d_model=128):
        super().__init__()
        if bucket_edges is None:
            bucket_edges = [0, 1, 2, 3, 4, 5, 10, 20, 60, 120]
        self.register_buffer("edges", torch.tensor(bucket_edges, dtype=torch.long))
        self.n_buckets = len(bucket_edges) + 1
        self.embed = nn.Embedding(self.n_buckets + 1, bucket_dim, padding_idx=0)
        self.proj = nn.Linear(bucket_dim, d_model) if bucket_dim != d_model else nn.Identity()

    def forward(self, time_since_update):
        idx = torch.bucketize(time_since_update.long(), self.edges) + 1
        idx = idx.clamp(0, self.n_buckets)
        return self.proj(self.embed(idx))


class EmbeddingLayer(nn.Module):
    """
    Full token embedding:
      token = value_proj(value)
            + dual_embed(company_id, metric_id)
            + calendar_embed(...)
            + time_bucket_embed(tsu)
    """

    def __init__(self, n_companies=5000, n_metrics=600, d_model=128):
        super().__init__()
        self.value_proj = ValueProjection(d_model)
        self.dual_embed = DualEmbedding(n_companies, n_metrics, d_model=d_model)
        self.calendar_embed = CalendarEmbedding(d_model)
        self.time_bucket_embed = TimeBucketEmbedding(d_model=d_model)

    def forward(self, values, company_ids, metric_ids,
                day, month, dow, year_offset, time_since_update):
        v = self.value_proj(values)
        e = self.dual_embed(company_ids, metric_ids)
        c = self.calendar_embed(day, month, dow, year_offset)
        t = self.time_bucket_embed(time_since_update)
        return v + e + c + t
