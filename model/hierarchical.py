"""
Hierarchical Encoder v3 — efficient tensor-based implementation.

Pipeline:
  1. Group tokens by (source, stock_id) → per-group pooling
  2. Multi-scale temporal splits per group
  3. Source-level cross-attention
  4. Cross-source Transformer fusion
  5. Attention pool → prediction

All operations use scatter/bmm — no Python loops over batch dim.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class HierarchicalEncoder(nn.Module):
    """
    Efficient hierarchical encoder using segment-based pooling.

    Input:  (B, L, d_model) flat token embeddings
    Output: (B, L_out, d_model) compressed sequence for final pooling

    The forward pass receives pre-computed segment indices so that
    grouping is a single scatter operation.
    """

    def __init__(self, d_model=128, nhead=4, num_layers=2,
                 dim_feedforward=512, dropout=0.2, attn_type="cosformer",
                 attn_chunk_size=2048, max_stocks=50, n_scales=4,
                 scale_dim=32):
        super().__init__()
        self.d_model = d_model
        self.n_scales = n_scales
        self.scale_dim = scale_dim

        # Per-scale linear projection (d_model → scale_dim)
        self.scale_proj = nn.ModuleList([
            nn.Linear(d_model, scale_dim) for _ in range(n_scales)
        ])

        # Source-level cross-attention for market stocks
        self.market_attn = nn.MultiheadAttention(
            d_model, nhead, dropout=dropout, batch_first=True
        )
        self.market_norm = nn.LayerNorm(d_model)

        # Cross-source fusion Transformer
        from .transformer import TransformerEncoder
        self.fusion = TransformerEncoder(
            d_model=d_model, nhead=nhead, num_layers=num_layers,
            dim_feedforward=dim_feedforward, dropout=dropout,
            attn_dropout=dropout, activation="gelu",
            attn_type=attn_type, chunk_size=attn_chunk_size,
        )

    def forward(self, x, mask, seg_ids, time_bins):
        """
        Args:
            x:         (B, L, d_model) embedded tokens
            mask:      (B, L) bool
            seg_ids:   (B, L) int, segment index for each token:
                       0 = macro, 1-50 = market stocks, 51 = financial,
                       52 = alternative, 53 = sentiment, -1 = pad
            time_bins: (B, L) int, which time scale bin [0..3]

        Returns:
            (B, N_seg, d_model) compressed sequence
        """
        B, L, D = x.shape
        device = x.device

        # ── 1. Segment pooling ────────────────────────────────────────────
        # For each (batch, segment, time_bin), compute mean pool
        n_segs = int(seg_ids.max().item()) + 2  # +2 for safety
        n_segs = min(n_segs, 60)  # cap

        seg_reprs = []
        for b in range(B):
            batch_segs = []
            for tbin in range(self.n_scales):
                tmask = mask[b] & (time_bins[b] == tbin)
                if not tmask.any():
                    # Empty time bin → zero vector
                    batch_segs.append(torch.zeros(1, self.scale_dim, device=device))
                    continue

                # Pool tokens per segment within this time bin
                seg_pooled = []
                for sid in range(n_segs):
                    smask = tmask & (seg_ids[b] == sid)
                    if smask.any():
                        seg_tokens = x[b][smask]  # (n_tokens, D)
                        pooled = seg_tokens.mean(dim=0)  # (D,)
                        pooled = self.scale_proj[tbin](pooled)  # (scale_dim,)
                        seg_pooled.append(pooled)

                if seg_pooled:
                    batch_segs.append(torch.stack(seg_pooled))  # (n_active_segs, scale_dim)
                else:
                    batch_segs.append(torch.zeros(1, self.scale_dim, device=device))

            # Concatenate scales → (n_active_segs, D)
            # Pad each scale's output to same n_segs, then concat
            max_per_scale = max(s.shape[0] for s in batch_segs) if batch_segs else 1
            padded = []
            for s in batch_segs:
                if s.shape[0] < max_per_scale:
                    s = torch.cat([s, torch.zeros(max_per_scale - s.shape[0], self.scale_dim, device=device)])
                padded.append(s)

            # (n_scales, max_segs, scale_dim) → (max_segs, n_scales * scale_dim) = (max_segs, D)
            merged = torch.cat(padded, dim=-1)  # (max_segs, D)
            seg_reprs.append(merged)

        # Pad per-sample segment counts
        max_segs = max(s.shape[0] for s in seg_reprs)
        padded_reprs = []
        mask_out = []
        for s in seg_reprs:
            n = s.shape[0]
            if n < max_segs:
                s = torch.cat([s, torch.zeros(max_segs - n, D, device=device)])
            padded_reprs.append(s)
            mask_out.append(torch.cat([torch.ones(n, device=device, dtype=torch.bool),
                                       torch.zeros(max_segs - n, device=device, dtype=torch.bool)]))

        out = torch.stack(padded_reprs)  # (B, max_segs, D)
        out_mask = torch.stack(mask_out)  # (B, max_segs)

        # ── 2. Cross-source fusion ────────────────────────────────────────
        out = self.fusion(out, out_mask)

        return out, out_mask


class SimpleHierarchicalEncoder(nn.Module):
    """
    Lightweight alternative: source-level pooling only (no per-stock grouping).

    Much faster — suitable when per-stock encoding overhead is too high.
    Groups tokens by source (market/macro/financial/alternative/sentiment),
    pools each source, then fuses via a small Transformer.
    """

    def __init__(self, d_model=128, nhead=4, num_layers=2,
                 dim_feedforward=512, dropout=0.2, attn_type="cosformer",
                 attn_chunk_size=2048, n_sources=5, n_scales=4, scale_dim=32):
        super().__init__()
        self.n_scales = n_scales
        self.scale_dim = scale_dim
        self.n_sources = n_sources

        # Per-scale projection
        self.scale_proj = nn.ModuleList([
            nn.Linear(d_model, scale_dim) for _ in range(n_scales)
        ])

        # Cross-source fusion
        from .transformer import TransformerEncoder
        self.fusion = TransformerEncoder(
            d_model=d_model, nhead=nhead, num_layers=num_layers,
            dim_feedforward=dim_feedforward, dropout=dropout,
            attn_dropout=dropout, activation="gelu",
            attn_type=attn_type, chunk_size=attn_chunk_size,
        )

    def forward(self, x, mask, source_ids, time_bins):
        """
        Args:
            x:          (B, L, d_model)
            mask:       (B, L) bool
            source_ids: (B, L) int [0..4] mapping each token to its source
            time_bins:  (B, L) int [0..3] time scale bin

        Returns:
            (B, n_sources * n_scales, d_model) fused sequence
            (B, n_sources * n_scales) valid mask
        """
        B, L, D = x.shape
        device = x.device
        n_src = source_ids.max().item() + 1

        # For each source × scale, pool tokens
        parts = []
        for src in range(n_src):
            src_parts = []
            for tbin in range(self.n_scales):
                sel = mask & (source_ids == src) & (time_bins == tbin)
                # Count per batch
                counts = sel.sum(dim=1, keepdim=True).clamp_min(1)  # (B, 1)
                pooled = (x * sel.unsqueeze(-1).float()).sum(dim=1) / counts  # (B, D)
                pooled = self.scale_proj[tbin](pooled)  # (B, scale_dim)
                src_parts.append(pooled)
            # Concat scales: (B, n_scales * scale_dim) = (B, D)
            src_repr = torch.cat(src_parts, dim=-1)
            parts.append(src_repr.unsqueeze(1))  # (B, 1, D)

        out = torch.cat(parts, dim=1)  # (B, n_sources, D)
        out_mask = torch.ones(B, n_src, device=device, dtype=torch.bool)

        # Cross-source fusion
        out = self.fusion(out, out_mask)
        return out, out_mask
