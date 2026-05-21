"""
Comprehensive diagnosis: label bias, data leakage, time split, Lasso baseline.
"""
import sys; sys.path.insert(0,'.')
import pandas as pd
import numpy as np
import torch
from pathlib import Path

print("=" * 65)
print("  POST-TRAINING DIAGNOSTIC REPORT")
print("=" * 65)

# ── 1. Label direction distribution ──────────────────────────────
print("\n1. LABEL DIRECTION BIAS")
labels = pd.read_parquet("D:/financial_data/processed/labels.parquet")
pos_pct = (labels["label"] > 0).mean() * 100
neg_pct = (labels["label"] < 0).mean() * 100
print(f"  Labels > 0: {pos_pct:.1f}%")
print(f"  Labels < 0: {neg_pct:.1f}%")
print(f"  Mean: {labels['label'].mean():.4f}")
print(f"  ACF(1): {labels['label'].autocorr(lag=1):.4f}")
print(f"  ACF(5): {labels['label'].autocorr(lag=5):.4f}")
print(f"  ACF(21): {labels['label'].autocorr(lag=21):.4f}")

if abs(pos_pct - 50) < 5:
    print("  -> Direction balanced, model NOT just predicting one sign")
else:
    print(f"  -> Direction skewed ({pos_pct:.0f}/{neg_pct:.0f}) — check if model biased")

# ── 2. Model prediction bias check ──────────────────────────────
print("\n2. MODEL PREDICTION CHECK")
from utils.dataset import SlidingWindowDataset, collate_fn
from model import FinancialMoETransformer
from torch.utils.data import DataLoader, Subset
import torch

ds = SlidingWindowDataset(max_seq_len=8192, use_cache=False)
train_dates, val_dates, _ = ds.train_val_test_split()
val_idx = [i for i, d in enumerate(ds.window_dates) if d in val_dates]
train_idx = [i for i, d in enumerate(ds.window_dates) if d in train_dates]

# Match the architecture used during training
from config import model_cfg
model = FinancialMoETransformer(
    n_companies=ds.n_companies+2, n_metrics=ds.n_metrics+2,
    d_model=128, nhead=4, num_layers=model_cfg.num_layers,
    dim_feedforward=model_cfg.dim_feedforward,
    num_experts=model_cfg.num_experts, expert_hidden=model_cfg.expert_hidden,
)
ckpt = torch.load("checkpoints/best.pt", map_location="cpu", weights_only=False)
model.load_state_dict(ckpt["model"])
model.eval()

# Get all val predictions
all_preds, all_labels, all_dates = [], [], []
loader = DataLoader(Subset(ds, val_idx), batch_size=2, shuffle=False,
                    collate_fn=collate_fn, num_workers=0, pin_memory=False)
with torch.no_grad():
    for batch in loader:
        if not batch or 'values' not in batch: continue
        pred, _ = model(
            batch['values'], batch['company_ids'], batch['metric_ids'],
            batch['day'], batch['month'], batch['dow'],
            batch['year_offset'], batch['time_since_update'],
            batch['mask'].bool(), batch['source_ids'], batch['time_bins'],
        )
        all_preds.extend(pred.numpy().flatten())
        all_labels.extend(batch['label'].numpy().flatten())

preds = np.array(all_preds)
lbls = np.array(all_labels)
print(f"  Val samples: {len(preds)}")
print(f"  Pred mean: {preds.mean():.4f}, std: {preds.std():.4f}")
print(f"  Pred > 0: {(preds > 0).mean()*100:.1f}%")
print(f"  Pred < 0: {(preds < 0).mean()*100:.1f}%")
print(f"  Pred same sign always: {'YES' if (preds > 0).all() or (preds < 0).all() else 'NO'}")
print(f"  IC: {np.corrcoef(preds, lbls)[0,1]:.4f}")

if np.abs(preds).max() < 0.01:
    print("  -> Model predictions near zero — underfitting")

# ── 3. Data leakage check ───────────────────────────────────────
print("\n3. DATA LEAKAGE CHECK")

dp = pd.read_parquet("D:/financial_data/processed/data_points.parquet")
# Check if any feature has suspiciously high correlation with label
# Sample: check close price vs future label
market_close = dp[(dp["source"] == "market") & (dp["variable"] == "close")]

if len(market_close) > 0:
    merged = market_close[["datetime", "value"]].merge(
        labels, on="datetime", how="inner"
    )
    if len(merged) > 100:
        # Check if current close correlates with future label (should be near 0)
        ic_close = merged["value"].corr(merged["label"])
        ic_abs = merged["value"].abs().corr(merged["label"])
        print(f"  IC(close_t, label_{t+21}): {ic_close:.4f}")
        print(f"  IC(|close_t|, label_{t+21}): {ic_abs:.4f}")
        if abs(ic_close) > 0.1:
            print(f"  WARNING: Unexpected high correlation — possible leakage")
        else:
            print(f"  -> OK: near-zero correlation, no obvious leakage")

# ── 4. Time split verification ──────────────────────────────────
print("\n4. TIME SPLIT VERIFICATION")
train_dates, val_dates, test_dates = ds.train_val_test_split()
train_dt = pd.to_datetime(train_dates)
val_dt = pd.to_datetime(val_dates)
test_dt = pd.to_datetime(test_dates)
print(f"  Train: {train_dt.min().date()} -> {train_dt.max().date()} ({len(train_dates)} windows)")
print(f"  Val:   {val_dt.min().date()} -> {val_dt.max().date()} ({len(val_dates)} windows)")
print(f"  Test:  {test_dt.min().date()} -> {test_dt.max().date()} ({len(test_dates)} windows)")

# Check if val range is AFTER train range (correct chronological split)
val_after_train = val_dt.min() > train_dt.max()
print(f"  Chronological split: {'YES' if val_after_train else 'WARNING: overlap!'}")

# ── 5. Lasso baseline ───────────────────────────────────────────
print("\n5. LASSO BASELINE (quick signal check)")
from sklearn.linear_model import LassoCV
from sklearn.preprocessing import StandardScaler

# Build a simplified feature matrix from the training windows
# Use mean-pooled per-source features for speed
X_list, y_list = [], []
loader = DataLoader(Subset(ds, train_idx[:300]), batch_size=1, shuffle=False,
                    collate_fn=collate_fn, num_workers=0, pin_memory=False)

print("  Building feature matrix from 300 windows...")
for batch in loader:
    if not batch or 'values' not in batch: continue
    x = batch['values'][0][batch['mask'][0]].numpy().flatten()
    # Take first 100 values as features for speed
    if len(x) > 100:
        x = np.concatenate([
            x[:50],  # first 50 values
            [x.mean(), x.std(), np.percentile(x, 10), np.percentile(x, 90)],
        ])
    else:
        continue
    X_list.append(x)
    y_list.append(batch['label'].item())

if len(X_list) > 100:
    # Pad to same length
    max_len = max(len(x) for x in X_list)
    X = np.zeros((len(X_list), max_len))
    for i, x in enumerate(X_list):
        X[i, :len(x)] = x
    y = np.array(y_list)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    lasso = LassoCV(cv=5, max_iter=2000, random_state=42)
    lasso.fit(X_scaled, y)
    y_pred = lasso.predict(X_scaled)
    ic_lasso = np.corrcoef(y_pred, y)[0, 1]
    n_nonzero = (lasso.coef_ != 0).sum()
    print(f"  Lasso IC (in-sample): {ic_lasso:.4f}")
    print(f"  Non-zero coefficients: {n_nonzero}/{X.shape[1]}")
    if ic_lasso > 0.01:
        print(f"  -> POSITIVE signal detected! Data has weak but real predictability")
    else:
        print(f"  -> Near-zero IC even with Lasso — data signal extremely weak")
else:
    print(f"  Not enough samples ({len(X_list)}), skipping Lasso")

print("\n" + "=" * 65)
print("  DIAGNOSIS COMPLETE")
print("=" * 65)
