"""
Post-training analysis: IC curve + Expert utilization.
"""
import sys; sys.path.insert(0,'.')
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from utils.dataset import SlidingWindowDataset, collate_fn
from model import FinancialMoETransformer
from config import model_cfg

ds = SlidingWindowDataset(max_seq_len=8192, use_cache=False)
_, val_dates, test_dates = ds.train_val_test_split()
val_idx = [i for i, d in enumerate(ds.window_dates) if d in val_dates]
test_idx = [i for i, d in enumerate(ds.window_dates) if d in test_dates]

model = FinancialMoETransformer(
    n_companies=ds.n_companies+2, n_metrics=ds.n_metrics+2,
    d_model=128, nhead=4, num_layers=model_cfg.num_layers,
    dim_feedforward=model_cfg.dim_feedforward,
    num_experts=model_cfg.num_experts, expert_hidden=model_cfg.expert_hidden,
)
ckpt = torch.load("checkpoints/best.pt", map_location="cpu", weights_only=False)
model.load_state_dict(ckpt["model"])
model.eval()
print(f"Loaded best model (epoch {ckpt['epoch']+1}, IC={ckpt['best_ic']:.4f})")

# ── IC Curve ──────────────────────────────────────────────────
print("\n" + "=" * 55)
print("  IC CURVE (Training History)")
print("=" * 55)

v1_ic = [-0.0494, -0.0497, -0.0494, -0.0565, -0.0564, -0.0619, -0.0619, -0.0619]
v2_ic = [0.0028, 0.0029, 0.0037, 0.0090, 0.0159, 0.0081, 0.0081, 0.0113, 0.0114, 0.0113, 0.0114]

print("\n  V1 (89K vars, 21d horizon):")
for i, ic in enumerate(v1_ic):
    bar = "#" * int(abs(ic) * 200) if ic < 0 else "+" * int(ic * 200)
    print(f"    Ep {i+1:2d}: {ic:+7.4f}  {bar}")

print(f"\n  V2 (40 core metrics, 63d horizon):")
for i, ic in enumerate(v2_ic):
    bar = "+" * max(1, int(ic * 500))
    print(f"    Ep {i+2:2d}: {ic:+7.4f}  {bar}")

print(f"\n  Best V1: -0.0494  |  Best V2: +0.0159  |  Δ = +0.0653")

# ── Expert Utilization ─────────────────────────────────────────
print(f"\n{'='*55}")
print(f"  EXPERT UTILIZATION (Validation Set)")
print(f"{'='*55}")

# Collect expert weights on validation set
loader = DataLoader(Subset(ds, val_idx), batch_size=2, shuffle=False,
                    collate_fn=collate_fn, num_workers=0, pin_memory=False)

all_gate_weights = []
all_preds = []
all_labels = []
all_dates = []

with torch.no_grad():
    for i, batch in enumerate(loader):
        if not batch or 'values' not in batch: continue

        # Get expert weights
        x = model.embedding(
            batch['values'], batch['company_ids'], batch['metric_ids'],
            batch['day'], batch['month'], batch['dow'],
            batch['year_offset'], batch['time_since_update'],
        )
        x, hier_mask = model.hierarchical(
            x, batch['mask'].bool(), batch['source_ids'], batch['time_bins'],
        )
        x_pooled = model.pooling(x, hier_mask)
        gates = model.moe.get_expert_weights(x_pooled)  # (B, E)

        all_gate_weights.append(gates.numpy())

        # Predictions
        pred, _ = model(
            batch['values'], batch['company_ids'], batch['metric_ids'],
            batch['day'], batch['month'], batch['dow'],
            batch['year_offset'], batch['time_since_update'],
            batch['mask'].bool(), batch['source_ids'], batch['time_bins'],
        )
        all_preds.extend(pred.numpy().flatten())
        all_labels.extend(batch['label'].numpy().flatten())

gates = np.concatenate(all_gate_weights, axis=0)  # (N_samples, 6 experts)
preds = np.array(all_preds)
labels = np.array(all_labels)

print(f"\n  Expert Gate Weights — Mean ± Std:")
print(f"  {'Expert':<10s} {'Mean Weight':>12s} {'Std':>8s} {'Usage %':>8s} {'Active':>8s}")
print(f"  {'-'*10} {'-'*12} {'-'*8} {'-'*8} {'-'*8}")

# Top-1 usage: which expert is selected per sample
top1_idx = gates.argmax(axis=1)  # (N,)
usage_pct = [(top1_idx == e).mean() * 100 for e in range(6)]

for e in range(6):
    mean_w = gates[:, e].mean()
    std_w = gates[:, e].std()
    print(f"  Expert {e+1:<4} {mean_w:>11.4f}  {std_w:>7.4f}  {usage_pct[e]:>7.1f}%  "
          f"{'#' * int(usage_pct[e] / 2):<20s}")

# Load balance
usage_arr = np.array(usage_pct)
ideal = 100 / 6
cv = usage_arr.std() / usage_arr.mean()
print(f"\n  Load Balance:")
print(f"    Ideal:   {ideal:.1f}% per expert")
print(f"    Actual:  min={usage_arr.min():.1f}%  max={usage_arr.max():.1f}%")
print(f"    CV:      {cv:.3f}  (0 = perfect balance)")

# ── Gate weights over time ────────────────────────────────────
print(f"\n  Expert Weights by Sample (first 100 val samples):")
print(f"  Sample  Expert1  Expert2  Expert3  Expert4  Expert5  Expert6  |  Active")
print(f"  {'-'*60}")
for i in range(min(20, len(gates))):
    top = gates[i].argmax()
    w_str = " ".join(f"{w:.3f}" for w in gates[i])
    marker = " " * (top * 8) + "▲"
    print(f"  {i:4d}    {w_str}  |  {top+1}")

# ── Val IC breakdown by expert ────────────────────────────────
print(f"\n  Val IC by Dominant Expert:")
for e in range(6):
    mask_e = top1_idx == e
    if mask_e.sum() > 5:
        ic_e = np.corrcoef(preds[mask_e], labels[mask_e])[0, 1] if mask_e.sum() > 1 else 0
        print(f"    Expert {e+1}: IC={ic_e:.4f}  (n={mask_e.sum():,} samples)")

# ── Overall Val IC ────────────────────────────────────────────
val_ic = np.corrcoef(preds, labels)[0, 1]
print(f"\n  Overall Val IC: {val_ic:.4f}")
