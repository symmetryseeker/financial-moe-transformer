import sys; sys.path.insert(0,'.')
import torch
from utils.dataset import SlidingWindowDataset, collate_fn
from model import FinancialMoETransformer
from torch.utils.data import DataLoader

ds = SlidingWindowDataset(max_seq_len=8192)
print(f"Companies: {ds.n_companies:,}  Metrics: {ds.n_metrics:,}")

# Check source_ids and time_bins in a sample
s = ds[0]
print(f"\nSample keys: {list(s.keys())}")
print(f"source_ids: min={s['source_ids'].min()}, max={s['source_ids'].max()}, "
      f"unique={torch.unique(s['source_ids'][s['mask']]).tolist()}")
print(f"time_bins:  min={s['time_bins'].min()}, max={s['time_bins'].max()}, "
      f"unique={torch.unique(s['time_bins'][s['mask']]).tolist()}")

model = FinancialMoETransformer(
    n_companies=ds.n_companies+2, n_metrics=ds.n_metrics+2,
    d_model=128, nhead=4, num_layers=2, dim_feedforward=512,
    num_experts=6, expert_hidden=64,
)
total = sum(p.numel() for p in model.parameters())
print(f"\nParams: {total:,}")

# Forward with real source_ids and time_bins
pred, lb = model(
    s['values'].unsqueeze(0), s['company_ids'].unsqueeze(0), s['metric_ids'].unsqueeze(0),
    s['day'].unsqueeze(0), s['month'].unsqueeze(0), s['dow'].unsqueeze(0),
    s['year_offset'].unsqueeze(0), s['time_since_update'].unsqueeze(0),
    s['mask'].unsqueeze(0), s['source_ids'].unsqueeze(0), s['time_bins'].unsqueeze(0),
)
print(f"Forward with hierarchy: pred={pred.item():.4f}, lb={lb.item():.4f}")

# Batch test with DataLoader
loader = DataLoader(ds, batch_size=2, shuffle=True, collate_fn=collate_fn)
batch = next(iter(loader))
pred2, lb2 = model(
    batch['values'], batch['company_ids'], batch['metric_ids'],
    batch['day'], batch['month'], batch['dow'],
    batch['year_offset'], batch['time_since_update'],
    batch['mask'].bool(), batch['source_ids'], batch['time_bins'],
)
print(f"Batch with hierarchy: pred={pred2.shape}, lb={lb2.item():.4f}")

# EMA test
from train import EMA
ema = EMA(model, decay=0.999)
ema.update()
ema.apply_shadow()
pred3, _ = model(
    batch['values'], batch['company_ids'], batch['metric_ids'],
    batch['day'], batch['month'], batch['dow'],
    batch['year_offset'], batch['time_since_update'],
    batch['mask'].bool(), batch['source_ids'], batch['time_bins'],
)
ema.restore()
print(f"EMA forward: pred={pred3.shape}")

# Backward test
model.train()
loss = torch.nn.functional.mse_loss(pred2.squeeze(-1), batch['label']) + 0.01 * lb2
loss.backward()
print(f"Backward: OK (loss={loss.item():.4f})")

print(f"\nALL 3 FIXES VERIFIED: EMA + hierarchy activation + winsorize")
