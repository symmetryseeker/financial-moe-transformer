"""
Transformer encoder with chunked linear attention (cosFormer).

Supports very long sequences (10K-100K tokens) by processing in chunks:
  K^T V is accumulated chunk-by-chunk, then Q * (accumulated KV).
  This gives O(L·d²) time AND O(chunk_size·d) memory.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


# ── Kernels ───────────────────────────────────────────────────────────────────

def elu_kernel(x: torch.Tensor) -> torch.Tensor:
    return F.elu(x) + 1.0

def relu_kernel(x: torch.Tensor) -> torch.Tensor:
    return F.relu(x)


# ── Chunked cosFormer Attention ───────────────────────────────────────────────

def chunked_cosformer_attention(query: torch.Tensor, key: torch.Tensor,
                                 value: torch.Tensor, mask: torch.Tensor = None,
                                 chunk_size: int = 2048) -> torch.Tensor:
    """
    Chunked cosFormer: process in chunks of `chunk_size` to limit peak memory.

    Algorithm:
      For each chunk c:
        Q_c = relu(Q_c) * cos_weight_c
        K_c = relu(K_c) * cos_weight_c
        if causal:  (we use bidirectional)
          update KV += K_c^T V_c
          out_c = Q_c * KV / (Q_c * sum(K_c^T))
        else:
          # Bidirectional: need full KV for each chunk
          # First pass: accumulate KV += K^T V for all chunks
          # Second pass: out_c = Q_c * KV / norm

    Args:
      query, key, value: (B, H, L, d_head)
      mask: (B, L) bool, True = valid
      chunk_size: tokens per chunk

    Returns:
      (B, H, L, d_head)
    """
    B, H, L, D = query.shape
    device = query.device

    # Cosine re-weighting
    t = torch.arange(L, device=device, dtype=query.dtype)
    cos_weight = torch.cos(math.pi / 2.0 * t / max(L - 1, 1))
    cos_weight = cos_weight.view(1, 1, L, 1)

    Q = relu_kernel(query) * cos_weight
    K = relu_kernel(key) * cos_weight
    V = value

    if mask is not None:
        m = mask.unsqueeze(1).unsqueeze(-1).to(Q.dtype)
        K = K * m
        V = V * m

    # First pass: accumulate global KV = sum(K^T V) over all chunks
    KV = torch.zeros(B, H, D, D, device=device, dtype=Q.dtype)
    for start in range(0, L, chunk_size):
        end = min(start + chunk_size, L)
        K_chunk = K[:, :, start:end, :]   # (B, H, cs, D)
        V_chunk = V[:, :, start:end, :]   # (B, H, cs, D)
        KV += torch.einsum("bhld,bhlf->bhdf", K_chunk, V_chunk)

    # Second pass: compute output chunk by chunk
    out_chunks = []
    for start in range(0, L, chunk_size):
        end = min(start + chunk_size, L)
        Q_chunk = Q[:, :, start:end, :]   # (B, H, cs, D)
        K_chunk = K[:, :, start:end, :]

        # Q_chunk * KV
        QKV_chunk = torch.einsum("bhld,bhdf->bhlf", Q_chunk, KV)

        # Normalizer (per chunk, based on K sum)
        Z = K.sum(dim=2).unsqueeze(-1).clamp_min(1e-8)  # (B, H, D, 1)
        QZ_chunk = torch.einsum("bhld,bhdf->bhlf", Q_chunk, Z).clamp_min(1e-8)

        out_chunk = QKV_chunk / QZ_chunk
        out_chunks.append(out_chunk)

    return torch.cat(out_chunks, dim=2)  # (B, H, L, D)


# ── Simple linear attention (fallback) ────────────────────────────────────────

def linear_attention(query, key, value, mask=None, kernel_fn=elu_kernel):
    B, H, L, D = query.shape
    Q = kernel_fn(query)
    K = kernel_fn(key)
    V = value
    if mask is not None:
        m = mask.unsqueeze(1).unsqueeze(-1).to(Q.dtype)
        K = K * m
        V = V * m
    KV = torch.einsum("bhld,bhlf->bhdf", K, V)
    Z = K.sum(dim=2).unsqueeze(-1).clamp_min(1e-8)
    QKV = torch.einsum("bhld,bhdf->bhlf", Q, KV)
    QZ = torch.einsum("bhld,bhdf->bhlf", Q, Z).clamp_min(1e-8)
    return QKV / QZ


# ── Multi-head Linear Attention ───────────────────────────────────────────────

class MultiheadLinearAttention(nn.Module):
    def __init__(self, d_model=128, nhead=4, dropout=0.2,
                 attn_type="cosformer", chunk_size=2048):
        super().__init__()
        assert d_model % nhead == 0
        self.d_model = d_model
        self.nhead = nhead
        self.d_head = d_model // nhead
        self.attn_type = attn_type
        self.chunk_size = chunk_size

        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

        self._reset_parameters()

    def _reset_parameters(self):
        for proj in [self.q_proj, self.k_proj, self.v_proj, self.out_proj]:
            nn.init.xavier_uniform_(proj.weight, gain=1.0 / math.sqrt(2.0))
            if proj.bias is not None:
                nn.init.zeros_(proj.bias)

    def _split_heads(self, x):
        B, L, D = x.shape
        return x.view(B, L, self.nhead, self.d_head).transpose(1, 2)

    def _merge_heads(self, x):
        B, H, L, D = x.shape
        return x.transpose(1, 2).contiguous().view(B, L, H * D)

    def forward(self, x, mask=None):
        Q = self._split_heads(self.q_proj(x))
        K = self._split_heads(self.k_proj(x))
        V = self._split_heads(self.v_proj(x))
        L = Q.shape[2]

        # Use chunked attention for long sequences
        if self.attn_type == "cosformer":
            if L > self.chunk_size * 2:
                attn_out = chunked_cosformer_attention(
                    Q, K, V, mask, chunk_size=self.chunk_size
                )
            else:
                # Inline cosFormer (no chunking needed for short sequences)
                t = torch.arange(L, device=Q.device, dtype=Q.dtype)
                cos_weight = torch.cos(math.pi / 2.0 * t / max(L - 1, 1))
                cos_weight = cos_weight.view(1, 1, L, 1)
                Qc = relu_kernel(Q) * cos_weight
                Kc = relu_kernel(K) * cos_weight
                Vc = V
                if mask is not None:
                    m = mask.unsqueeze(1).unsqueeze(-1).to(Qc.dtype)
                    Kc = Kc * m
                    Vc = Vc * m
                KV = torch.einsum("bhld,bhlf->bhdf", Kc, Vc)
                Z = Kc.sum(dim=2).unsqueeze(-1).clamp_min(1e-8)
                QKV = torch.einsum("bhld,bhdf->bhlf", Qc, KV)
                QZ = torch.einsum("bhld,bhdf->bhlf", Qc, Z).clamp_min(1e-8)
                attn_out = QKV / QZ
        else:
            attn_out = linear_attention(Q, K, V, mask, kernel_fn=elu_kernel)

        out = self._merge_heads(attn_out)
        return self.dropout(self.out_proj(out))


# ── Transformer Encoder Layer ─────────────────────────────────────────────────

class TransformerEncoderLayer(nn.Module):
    """Pre-LN transformer layer with chunked linear attention."""

    def __init__(self, d_model=128, nhead=4, dim_feedforward=512,
                 dropout=0.2, attn_dropout=0.2, activation="gelu",
                 attn_type="cosformer", chunk_size=2048):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.self_attn = MultiheadLinearAttention(
            d_model, nhead, attn_dropout, attn_type, chunk_size
        )
        self.dropout1 = nn.Dropout(dropout)

        self.norm2 = nn.LayerNorm(d_model)
        act_fn = nn.GELU() if activation == "gelu" else nn.ReLU()
        self.ffn = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            act_fn,
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model),
        )
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        # Self-attention (Pre-LN)
        x = x + self.dropout1(self.self_attn(self.norm1(x), mask))
        # FFN (Pre-LN)
        x = x + self.dropout2(self.ffn(self.norm2(x)))
        return x


# ── Transformer Encoder ───────────────────────────────────────────────────────

class TransformerEncoder(nn.Module):
    """Stack of encoder layers with final LayerNorm."""

    def __init__(self, d_model=128, nhead=4, num_layers=4, dim_feedforward=512,
                 dropout=0.2, attn_dropout=0.2, activation="gelu",
                 attn_type="cosformer", chunk_size=2048):
        super().__init__()
        self.layers = nn.ModuleList([
            TransformerEncoderLayer(
                d_model, nhead, dim_feedforward,
                dropout, attn_dropout, activation, attn_type, chunk_size
            )
            for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x, mask=None):
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)
