"""Debug: time a few actual training batches."""
import sys, time; sys.path.insert(0,'.')
import torch
from torch.utils.data import DataLoader
from utils.dataset import SlidingWindowDataset, collate_fn
from model import FinancialMoETransformer
from utils.losses import CombinedLoss

ds = SlidingWindowDataset(max_seq_len=8192, use_cache=False)
model = FinancialMoETransformer(
    n_companies=ds.n_companies+2, n_metrics=ds.n_metrics+2,
    d_model=128, nhead=4, num_layers=2, dim_feedforward=512,
    num_experts=6, expert_hidden=64,
)
loss_fn = CombinedLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)

loader = DataLoader(ds, batch_size=2, shuffle=True, collate_fn=collate_fn,
                    num_workers=0, pin_memory=False)

print("Timing 5 training batches...")
model.train()
for step, batch in enumerate(loader):
    if step >= 5: break
    t0 = time.time()
    optimizer.zero_grad()
    pred, lb = model(
        batch['values'], batch['company_ids'], batch['metric_ids'],
        batch['day'], batch['month'], batch['dow'],
        batch['year_offset'], batch['time_since_update'],
        batch['mask'].bool(), batch['source_ids'], batch['time_bins'],
    )
    loss, ld = loss_fn(pred, batch['label'], lb)
    loss.backward()
    optimizer.step()
    elapsed = time.time() - t0
    print(f"  batch {step}: {elapsed:.1f}s | MSE={ld['mse']:.3f} Rank={ld['rank']:.3f}", flush=True)

# Estimate full epoch time
print(f"\nEstimate: {len(loader)} batches/epoch × avg_time = ~{len(loader)*elapsed/60:.0f} min/epoch")
