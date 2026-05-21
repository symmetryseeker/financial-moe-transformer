"""Verify v2 model with updated data and architecture."""
import sys; sys.path.insert(0, '.')
import torch
from torch.utils.data import DataLoader
from utils.dataset import SlidingWindowDataset, collate_fn
from model import FinancialMoETransformer
from config import model_cfg

print("Loading dataset...")
ds = SlidingWindowDataset(
    data_path='data/processed/data_points.parquet',
    labels_path='data/processed/labels.parquet',
    window_days=365, forecast_horizon=21,
    max_seq_len=model_cfg.max_seq_len, use_cache=False
)
print(f"Windows: {len(ds)}, Vocab: {len(ds.var_to_id)}")

# Check sequence lengths
import numpy as np
seq_lens = []
for i in range(min(200, len(ds))):
    s = ds[i]
    if s is not None:
        seq_lens.append(s["seq_len"])
seq_lens = np.array(seq_lens)
print(f"Seq lens: mean={seq_lens.mean():.0f}, median={np.median(seq_lens):.0f}, "
      f"max={seq_lens.max()}, P90={np.percentile(seq_lens, 90):.0f}")

print("\nBuilding v2 model...")
model = FinancialMoETransformer(
    vocab_size=max(model_cfg.vocab_size, len(ds.var_to_id) + 2),
    d_model=model_cfg.d_model, nhead=model_cfg.nhead,
    num_layers=model_cfg.num_layers,
    dim_feedforward=model_cfg.dim_feedforward,
    attn_type=model_cfg.attn_type,
    num_experts=model_cfg.num_experts,
    expert_hidden=model_cfg.expert_hidden,
    top_k=model_cfg.top_k,
    max_seq_len=model_cfg.max_seq_len,
    attn_chunk_size=model_cfg.attn_chunk_size,
)
n = sum(p.numel() for p in model.parameters())
print(f"Parameters: {n:,}")

# Break down by module
for name, mod in [("Embedding", model.embedding), ("Encoder", model.encoder),
                   ("Pooling", model.pooling), ("MoE", model.moe)]:
    n_mod = sum(p.numel() for p in mod.parameters())
    print(f"  {name}: {n_mod:,}")

# Single forward pass
print("\nTesting forward pass...")
sample = ds[0]
batch = {k: v.unsqueeze(0) for k, v in sample.items()
         if isinstance(v, torch.Tensor)}
pred, lb = model(**{k: v for k, v in batch.items() if k != "label" and k != "seq_len"})
print(f"  pred={pred.item():.4f}, lb_loss={lb.item():.4f}")

# Batch forward
print("Testing batch...")
loader = DataLoader(ds, batch_size=2, shuffle=True, collate_fn=collate_fn)
batch2 = next(iter(loader))
pred2, lb2 = model(
    batch2["values"], batch2["var_ids"], batch2["day"], batch2["month"],
    batch2["dow"], batch2["year_offset"], batch2["time_since_update"],
    batch2["mask"].bool()
)
mse = torch.nn.functional.mse_loss(pred2.squeeze(-1), batch2["label"])
print(f"  MSE={mse.item():.4f}, LB={lb2.item():.4f}")

# Training step
print("Testing backward + optimizer step...")
model.train()
optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
loss = mse + 0.01 * lb2
loss.backward()
optimizer.step()
optimizer.zero_grad()
print(f"  Loss={loss.item():.4f}, Gradients OK")

print(f"\n{'='*50}")
print(f"  V2 MODEL VERIFIED — {n:,} params, {len(ds)} windows")
print(f"  Ready for training")
print(f"{'='*50}")
