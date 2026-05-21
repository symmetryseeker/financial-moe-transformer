import sys; sys.path.insert(0,'.')
from utils.dataset import SlidingWindowDataset
from model import FinancialMoETransformer
import torch

ds = SlidingWindowDataset(max_seq_len=8192)
print(f"Companies: {ds.n_companies:,}  Metrics: {ds.n_metrics:,}")

model = FinancialMoETransformer(
    n_companies=ds.n_companies+2, n_metrics=ds.n_metrics+2,
    d_model=128, nhead=4, num_layers=2, dim_feedforward=512,
    num_experts=6, expert_hidden=64,
)
total = sum(p.numel() for p in model.parameters())
print(f"Params: {total:,} ({'OK' if total < 2e6 else 'OVER 2M!'})")

# Breakdown
for name, mod in [("Embedding", model.embedding), ("Hierarchical", model.hierarchical),
                   ("Pooling", model.pooling), ("MoE", model.moe)]:
    n = sum(p.numel() for p in mod.parameters())
    print(f"  {name}: {n:,} ({n/total*100:.0f}%)")

# Forward
s = ds[0]
pred, lb = model(
    s['values'].unsqueeze(0), s['company_ids'].unsqueeze(0), s['metric_ids'].unsqueeze(0),
    s['day'].unsqueeze(0), s['month'].unsqueeze(0), s['dow'].unsqueeze(0),
    s['year_offset'].unsqueeze(0), s['time_since_update'].unsqueeze(0), s['mask'].unsqueeze(0),
)
print(f"Forward: OK (pred={pred.item():.4f}, lb={lb.item():.4f})")
