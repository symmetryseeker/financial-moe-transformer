"""Quick diagnosis: Lasso baseline + data quality checks."""
import sys; sys.path.insert(0,'.')
import pandas as pd
import numpy as np
from utils.dataset import SlidingWindowDataset, collate_fn
from torch.utils.data import DataLoader, Subset
import torch

print("=" * 55)
print("  QUICK DIAGNOSIS: Signal Check + Lasso Baseline")
print("=" * 55)

# ── Label stats ────────────────────────────────────────────────
labels = pd.read_parquet("D:/financial_data/processed/labels.parquet")
pos_pct = (labels["label"] > 0).mean() * 100
print(f"\nLabel: >0={pos_pct:.1f}%  mean={labels['label'].mean():.4f}  std={labels['label'].std():.4f}")
print(f"ACF(1)={labels['label'].autocorr(1):.4f}  ACF(21)={labels['label'].autocorr(21):.4f}")

# ── Time split ─────────────────────────────────────────────────
ds = SlidingWindowDataset(max_seq_len=8192, use_cache=False)
td, vd, tsd = ds.train_val_test_split()
td = pd.to_datetime(td); vd = pd.to_datetime(vd); tsd = pd.to_datetime(tsd)
print(f"\nTime split: Train [{td.min().date()} -> {td.max().date()}] {len(td)}w")
print(f"            Val   [{vd.min().date()} -> {vd.max().date()}] {len(vd)}w")
print(f"            Test  [{tsd.min().date()} -> {tsd.max().date()}] {len(tsd)}w")
print(f"Chronological: {'YES' if vd.min() > td.max() else 'OVERLAP!'}")

# ── Data leakage ───────────────────────────────────────────────
dp = pd.read_parquet("D:/financial_data/processed/data_points.parquet",
                     columns=["datetime", "source", "variable", "value"])
market_c = dp[(dp["source"]=="market") & (dp["variable"]=="close")]
merged = market_c[["datetime","value"]].merge(labels, on="datetime", how="inner")
if len(merged) > 100:
    ic = merged["value"].corr(merged["label"])
    print(f"\nLeakage check: IC(close, label) = {ic:.4f}  {'OK' if abs(ic)<0.1 else 'WARN'}")

# ── Lasso baseline: 1 variable at a time ──────────────────────
print("\nSingle-variable IC scan (top 20):")
# Get unique variables
vars_list = dp["variable"].unique()[:500]  # sample 500 vars
train_dates_set = set(td)
var_ics = []
for v in vars_list[:200]:  # scan first 200
    vdf = dp[dp["variable"] == v][["datetime", "value"]]
    m = vdf.merge(labels, on="datetime", how="inner")
    m = m[m["datetime"].isin(train_dates_set)]
    if len(m) > 50:
        ic = m["value"].corr(m["label"])
        if abs(ic) > 0.02:
            var_ics.append((v, ic, len(m)))

var_ics.sort(key=lambda x: -abs(x[1]))
for v, ic, n in var_ics[:20]:
    print(f"  {str(v)[:55]:55s} IC={ic:+.4f}  n={n}")

if var_ics:
    top_ic = max(abs(x[1]) for x in var_ics)
    n_positive = sum(1 for x in var_ics if x[1] > 0.02)
    print(f"\n  Max |IC|: {top_ic:.4f} | vars with IC>0.02: {n_positive}/{len(var_ics)}")
else:
    print("  No variable with |IC| > 0.02 — signal extremely weak")

# ── Lasso: multi-variable ─────────────────────────────────────
print("\nLasso multi-variable baseline (300 samples):")
train_idx = [i for i, d in enumerate(ds.window_dates) if d in train_dates_set]
loader = DataLoader(Subset(ds, train_idx[:300]), batch_size=1, shuffle=False,
                    collate_fn=collate_fn, num_workers=0, pin_memory=False)

X_list, y_list = [], []
for batch in loader:
    if not batch or 'values' not in batch: continue
    vals = batch['values'][0][batch['mask'][0]].numpy().flatten()
    if len(vals) < 10: continue
    # Simple features
    features = [
        vals.mean(), vals.std(),
        np.percentile(vals, 10), np.percentile(vals, 25),
        np.percentile(vals, 50), np.percentile(vals, 75),
        np.percentile(vals, 90),
        (vals > 0).mean(), (vals > 1).mean(), (vals < -1).mean(),
    ]
    X_list.append(features)
    y_list.append(batch['label'].item())

if len(X_list) > 50:
    from sklearn.linear_model import LassoCV
    from sklearn.preprocessing import StandardScaler
    X = np.array(X_list); y = np.array(y_list)
    X = StandardScaler().fit_transform(X)
    lasso = LassoCV(cv=5, max_iter=3000, random_state=42).fit(X, y)
    pred = lasso.predict(X)
    ic_lasso = np.corrcoef(pred, y)[0, 1]
    print(f"  Lasso IC (in-sample, {len(y)} samples): {ic_lasso:.4f}")
    print(f"  Non-zero coefs: {(lasso.coef_!=0).sum()}/{X.shape[1]}")

print("\n" + "=" * 55)
print("  DONE")
