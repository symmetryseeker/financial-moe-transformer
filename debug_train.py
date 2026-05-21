"""Debug: profile training step to find bottleneck."""
import sys, time; sys.path.insert(0,'.')
import torch
from utils.dataset import SlidingWindowDataset, collate_fn
from model import FinancialMoETransformer
from torch.utils.data import DataLoader

ds = SlidingWindowDataset(max_seq_len=8192, use_cache=False)
print(f"Dataset: {len(ds)} windows")

model = FinancialMoETransformer(
    n_companies=ds.n_companies+2, n_metrics=ds.n_metrics+2,
    d_model=128, nhead=4, num_layers=2, dim_feedforward=512,
    num_experts=6, expert_hidden=64,
)
total = sum(p.numel() for p in model.parameters())
print(f"Params: {total:,}")

# Get one batch
loader = DataLoader(ds, batch_size=1, shuffle=True, collate_fn=collate_fn)
batch = next(iter(loader))
print(f"Batch: values={batch['values'].shape}, source_ids unique={torch.unique(batch['source_ids']).tolist()}")
print(f"time_bins unique={torch.unique(batch['time_bins']).tolist()}")

# Profile each step
steps = [
    ("Embedding", lambda: model.embedding(
        batch['values'], batch['company_ids'], batch['metric_ids'],
        batch['day'], batch['month'], batch['dow'],
        batch['year_offset'], batch['time_since_update'])),
    ("Hierarchical", lambda: model.hierarchical(
        model.embedding(batch['values'], batch['company_ids'], batch['metric_ids'],
                        batch['day'], batch['month'], batch['dow'],
                        batch['year_offset'], batch['time_since_update']),
        batch['mask'].bool(), batch['source_ids'], batch['time_bins'])),
]

for name, fn in steps:
    t0 = time.time()
    try:
        result = fn()
        elapsed = time.time() - t0
        if isinstance(result, tuple):
            print(f"{name}: {elapsed:.1f}s → shapes={[r.shape for r in result]}")
        else:
            print(f"{name}: {elapsed:.1f}s → shape={result.shape}")
    except Exception as e:
        elapsed = time.time() - t0
        print(f"{name}: FAIL after {elapsed:.1f}s — {e}")
        break

# Full forward pass
print("\nFull forward pass...")
t0 = time.time()
pred, lb = model(
    batch['values'], batch['company_ids'], batch['metric_ids'],
    batch['day'], batch['month'], batch['dow'],
    batch['year_offset'], batch['time_since_update'],
    batch['mask'].bool(), batch['source_ids'], batch['time_bins'],
)
print(f"Forward: {time.time()-t0:.1f}s, pred={pred.item():.4f}")

# Backward
print("Backward...")
t0 = time.time()
loss = torch.nn.functional.mse_loss(pred.squeeze(-1), batch['label']) + 0.01 * lb
loss.backward()
print(f"Backward: {time.time()-t0:.1f}s")
