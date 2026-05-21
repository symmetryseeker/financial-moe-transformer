"""Quick expert utilization analysis."""
import sys; sys.path.insert(0,'.')
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from utils.dataset import SlidingWindowDataset, collate_fn
from model import FinancialMoETransformer
from config import model_cfg

ds = SlidingWindowDataset(max_seq_len=8192, use_cache=False)
_, val_dates, _ = ds.train_val_test_split()
val_idx = [i for i, d in enumerate(ds.window_dates) if d in val_dates]

model = FinancialMoETransformer(
    n_companies=ds.n_companies+2, n_metrics=ds.n_metrics+2,
    d_model=128, nhead=4, num_layers=model_cfg.num_layers,
    dim_feedforward=model_cfg.dim_feedforward,
    num_experts=model_cfg.num_experts, expert_hidden=model_cfg.expert_hidden,
)
ckpt = torch.load("checkpoints/best.pt", map_location="cpu", weights_only=False)
model.load_state_dict(ckpt["model"])
model.eval()

# Fetch 100 val samples
loader = DataLoader(Subset(ds, val_idx[:100]), batch_size=1, shuffle=False,
                    collate_fn=collate_fn, num_workers=0, pin_memory=False)

all_gates = []
all_preds, all_labels = [], []
with torch.no_grad():
    for batch in loader:
        if not batch or 'values' not in batch: continue
        x = model.embedding(
            batch['values'], batch['company_ids'], batch['metric_ids'],
            batch['day'], batch['month'], batch['dow'],
            batch['year_offset'], batch['time_since_update'],
        )
        x, hm = model.hierarchical(x, batch['mask'].bool(),
                                   batch['source_ids'], batch['time_bins'])
        xp = model.pooling(x, hm)
        gates = model.moe.get_expert_weights(xp)
        all_gates.append(gates.numpy())
        pred, _ = model(
            batch['values'], batch['company_ids'], batch['metric_ids'],
            batch['day'], batch['month'], batch['dow'],
            batch['year_offset'], batch['time_since_update'],
            batch['mask'].bool(), batch['source_ids'], batch['time_bins'],
        )
        all_preds.extend(pred.numpy().flatten())
        all_labels.extend(batch['label'].numpy().flatten())

gates = np.concatenate(all_gates, axis=0)
top1 = gates.argmax(axis=1)
preds = np.array(all_preds)
labels = np.array(all_labels)

print("=" * 55)
print("  EXPERT UTILIZATION (100 Validation Samples)")
print("=" * 55)

print(f"\n  {'Expert':<10s} {'Mean Gate':>10s} {'Std':>8s} {'Top-1 %':>8s}  Distribution")
print(f"  {'-'*10} {'-'*10} {'-'*8} {'-'*8}  {'-'*20}")
for e in range(6):
    usage = (top1 == e).mean() * 100
    bar = "#" * int(usage)
    print(f"  Expert {e+1:<4} {gates[:,e].mean():>10.4f} {gates[:,e].std():>8.4f} {usage:>7.1f}%  {bar}")

usage_pct = [(top1 == e).mean() * 100 for e in range(6)]
ideal = 100 / 6
print(f"\n  Ideal: {ideal:.1f}% each  |  CV: {np.std(usage_pct)/np.mean(usage_pct):.3f}")

# IC by dominant expert
print(f"\n  IC by Dominant Expert:")
for e in range(6):
    mask = top1 == e
    if mask.sum() > 3:
        ic = np.corrcoef(preds[mask], labels[mask])[0,1] if mask.sum() > 1 else 0
        print(f"    Expert {e+1}: IC={ic:+.4f}  (n={mask.sum()})")

# Expert weight time series (first 30 samples)
print(f"\n  Gate Weights — First 15 Samples:")
print(f"  {'Sample':<8s} {'E1':>6s} {'E2':>6s} {'E3':>6s} {'E4':>6s} {'E5':>6s} {'E6':>6s}  {'Active':>6s}")
for i in range(min(15, len(gates))):
    ws = " ".join(f"{w:5.3f}" for w in gates[i])
    print(f"  {i:<8d} {ws}     Expert {top1[i]+1}")
