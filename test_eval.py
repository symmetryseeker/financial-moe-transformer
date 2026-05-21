"""
Evaluate best.pt on held-out TEST set (never seen during training).
"""
import sys; sys.path.insert(0,'.')
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Subset
from utils.dataset import SlidingWindowDataset, collate_fn
from model import FinancialMoETransformer
from config import model_cfg

ds = SlidingWindowDataset(max_seq_len=8192, use_cache=False)
train_dates, val_dates, test_dates = ds.train_val_test_split()
test_idx = [i for i, d in enumerate(ds.window_dates) if d in test_dates]

td = pd.to_datetime(train_dates); vd = pd.to_datetime(val_dates)
tsd = pd.to_datetime(test_dates)
print(f"Train: {td.min().date()} -> {td.max().date()}  ({len(train_dates)}w)")
print(f"Val:   {vd.min().date()} -> {vd.max().date()}  ({len(val_dates)}w)")
print(f"Test:  {tsd.min().date()} -> {tsd.max().date()}  ({len(test_dates)}w)")

model = FinancialMoETransformer(
    n_companies=ds.n_companies+2, n_metrics=ds.n_metrics+2,
    d_model=128, nhead=4, num_layers=model_cfg.num_layers,
    dim_feedforward=model_cfg.dim_feedforward,
    num_experts=model_cfg.num_experts, expert_hidden=model_cfg.expert_hidden,
)
ckpt = torch.load("checkpoints/best.pt", map_location="cpu", weights_only=False)
model.load_state_dict(ckpt["model"])
model.eval()
print(f"\nLoaded best model: epoch {ckpt['epoch']+1}, best val IC={ckpt['best_ic']:.4f}")

# ── Test set inference ──────────────────────────────────────────
loader = DataLoader(Subset(ds, test_idx), batch_size=2, shuffle=False,
                    collate_fn=collate_fn, num_workers=0, pin_memory=False)

all_preds, all_labels, all_gates = [], [], []
with torch.no_grad():
    for batch in loader:
        if not batch or 'values' not in batch: continue
        pred, _ = model(
            batch['values'], batch['company_ids'], batch['metric_ids'],
            batch['day'], batch['month'], batch['dow'],
            batch['year_offset'], batch['time_since_update'],
            batch['mask'].bool(), batch['source_ids'], batch['time_bins'],
        )
        # Expert weights
        x = model.embedding(
            batch['values'], batch['company_ids'], batch['metric_ids'],
            batch['day'], batch['month'], batch['dow'],
            batch['year_offset'], batch['time_since_update'],
        )
        x, hm = model.hierarchical(x, batch['mask'].bool(),
                                   batch['source_ids'], batch['time_bins'])
        xp = model.pooling(x, hm)
        gates = model.moe.get_expert_weights(xp)

        all_preds.extend(pred.numpy().flatten())
        all_labels.extend(batch['label'].numpy().flatten())
        all_gates.append(gates.numpy())

preds = np.array(all_preds)
labels = np.array(all_labels)
gates = np.concatenate(all_gates, axis=0)

# ── Metrics ─────────────────────────────────────────────────────
print(f"\n{'='*55}")
print(f"  TEST SET EVALUATION ({len(preds)} samples)")
print(f"{'='*55}")

ic = np.corrcoef(preds, labels)[0, 1]
rank_ic = np.corrcoef(preds.argsort().argsort(), labels.argsort().argsort())[0, 1]
mse = np.mean((preds - labels) ** 2)
mae = np.mean(np.abs(preds - labels))
r2 = 1 - mse / labels.var()

# Direction accuracy
dir_acc = (np.sign(preds) == np.sign(labels)).mean()

# Long-short: top 20% vs bottom 20%
n_long = max(1, len(preds) // 5)
top_idx = preds.argsort()[-n_long:]
bot_idx = preds.argsort()[:n_long]
long_ret = labels[top_idx].mean()
short_ret = labels[bot_idx].mean()
spread = long_ret - short_ret

# Sharpe (annualized, assuming 63-day returns)
spread_std = np.std(labels[top_idx]) / np.sqrt(n_long) + np.std(labels[bot_idx]) / np.sqrt(n_long)
sharpe = spread / max(spread_std, 1e-8) * np.sqrt(252 / 63)

print(f"  IC:           {ic:+.4f}")
print(f"  Rank IC:      {rank_ic:+.4f}")
print(f"  MSE:          {mse:.4f}")
print(f"  R²:           {r2:+.4f}")
print(f"  Direction Acc: {dir_acc:.1%}")
print(f"  Long-Short Spread: {spread:+.4f}")
print(f"  Sharpe (ann): {sharpe:+.2f}")

# ── Expert utilization on test ──────────────────────────────────
top1 = gates.argmax(axis=1)
print(f"\n  Expert Usage (Test Set):")
for e in range(6):
    usage = (top1 == e).mean() * 100
    ic_e = np.corrcoef(preds[top1 == e], labels[top1 == e])[0,1] if (top1==e).sum() > 3 else 0
    bar = "#" * int(usage)
    print(f"    Expert {e+1}: {usage:5.1f}%  IC={ic_e:+.4f}  {bar}")

# ── Comparison ──────────────────────────────────────────────────
print(f"\n  Summary:")
print(f"    Val IC (reported):  +0.0159")
print(f"    Test IC (now):      {ic:+.4f}")
print(f"    Signal confirmed:    {'YES' if ic > 0.005 else 'WEAK' if ic > -0.005 else 'NO'}")

if ic > 0.005:
    print(f"\n  -> Signal is REAL. Proceed with Phase 1.")
elif ic > -0.005:
    print(f"\n  -> Signal is WEAK (near zero). May need label/data review.")
else:
    print(f"\n  -> Signal is NEGATIVE. Stop and check for data leakage.")
