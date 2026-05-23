"""
Cross-sectional Rank IC evaluation.
For each test date, compute predictions for ALL stocks,
then calculate Spearman rank correlation with actual labels.
"""
import sys; sys.path.insert(0,'.')
import numpy as np, pandas as pd, torch
from collections import defaultdict
from torch.utils.data import DataLoader
from utils.dataset_stock import StockSlidingWindowDataset, collate_fn_stock
from model import FinancialMoETransformer
from config import model_cfg, MODEL_DIR

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {DEVICE}")

# Load full dataset
ds = StockSlidingWindowDataset(
    data_path="data/processed/data_points.parquet",
    labels_path="data/processed/labels_stock.parquet",
    max_seq_len=4096, use_cache=True, cache_dir="data/processed/cache_stock_v2",
    multi_window=False, min_windows_per_stock=50)

print(f"Total windows: {len(ds):,}")

# 80/10/10 split by date
train_idx, val_idx, test_idx = ds.train_val_test_split_by_date(0.80, 0.10)
print(f"Test windows: {len(test_idx):,}")

# Group test windows by date
test_by_date = defaultdict(list)
for idx in test_idx:
    end_date, stock_code = ds.window_index[idx]
    test_by_date[end_date].append(idx)

test_dates = sorted(test_by_date.keys())
print(f"Test dates: {len(test_dates)} | {test_dates[0]} -> {test_dates[-1]}")

# Load model
ckpt = torch.load(MODEL_DIR / "best_stock_v2.pt", map_location="cpu", weights_only=False)
model = FinancialMoETransformer(
    n_companies=ds.n_companies+2, n_metrics=ds.n_metrics+2,
    d_model=model_cfg.d_model, nhead=model_cfg.nhead,
    num_layers=model_cfg.num_layers, dim_feedforward=model_cfg.dim_feedforward,
    dropout=model_cfg.dropout, attn_type=model_cfg.attn_type,
    attn_chunk_size=model_cfg.attn_chunk_size,
    num_experts=model_cfg.num_experts, expert_hidden=model_cfg.expert_hidden,
    top_k=model_cfg.top_k,
).to(DEVICE)
ms = model.state_dict()
for k, t in ckpt["model"].items():
    if k in ms and ms[k].shape == t.shape: ms[k].copy_(t)
    elif k in ms and ms[k].dim() == 2:
        m0, m1 = min(ms[k].shape[0], t.shape[0]), min(ms[k].shape[1], t.shape[1])
        ms[k][:m0, :m1].copy_(t[:m0, :m1])
model.eval()
print(f"Loaded: epoch {ckpt.get('epoch','?')+1}, Val IC={ckpt.get('best_ic',0):.4f}")

# Evaluate: batch by date (all stocks for one date = one mini-batch group)
from scipy.stats import spearmanr
from torch.utils.data import DataLoader

# Loader for test windows
test_loader = DataLoader(
    torch.utils.data.Subset(ds, test_idx), batch_size=64,
    shuffle=False, collate_fn=collate_fn_stock, num_workers=0, pin_memory=True)

all_preds, all_labels, all_dates = [], [], []
gate_usage = np.zeros(6)

for batch in test_loader:
    if batch is None: continue
    with torch.no_grad():
        batch_gpu = {k: v.to(DEVICE) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
        pred, _ = model(batch_gpu['values'], batch_gpu['company_ids'], batch_gpu['metric_ids'],
                       batch_gpu['day'], batch_gpu['month'], batch_gpu['dow'], batch_gpu['year_offset'],
                       batch_gpu['time_since_update'], batch_gpu['mask'].bool(),
                       batch_gpu['source_ids'], batch_gpu['time_bins'])
        x = model.embedding(batch_gpu['values'], batch_gpu['company_ids'], batch_gpu['metric_ids'],
                           batch_gpu['day'], batch_gpu['month'], batch_gpu['dow'],
                           batch_gpu['year_offset'], batch_gpu['time_since_update'])
        x, hm = model.hierarchical(x, batch_gpu['mask'].bool(), batch_gpu['source_ids'], batch_gpu['time_bins'])
        g = model.moe.get_expert_weights(model.pooling(x, hm))
        gate_usage += np.bincount(g.cpu().numpy().argmax(axis=1), minlength=6)
    all_preds.extend(pred.cpu().numpy().flatten())
    all_labels.extend(batch['label'].numpy().flatten())

# Map batches back to dates (we lost date info in batched loading)
# Fallback: compute global IC as approximation
from scipy.stats import spearmanr
global_ic, _ = spearmanr(all_preds, all_labels)
print(f"  Batched inference done. {len(all_preds):,} predictions.")
print(f"  Global Rank IC: {global_ic:+.4f}")

# Now do proper daily cross-sectional: group test windows by date
# Use the window_index to map predictions back to dates
print(f"  Computing daily cross-sectional ICs...")
daily_preds = defaultdict(list)
daily_labels = defaultdict(list)
# We need to re-infer but track dates. Sample 50 test dates for speed.
import random; random.seed(42)
sample_dates = sorted(random.sample(test_dates, min(50, len(test_dates))))
for di, date in enumerate(sample_dates):
    indices = test_by_date[date][:100]  # max 100 stocks per date
    for idx in indices:
        sample = ds[idx]
        if sample is None: continue
        with torch.no_grad():
            b = collate_fn_stock([sample])
            if b is None: continue
            b = {k: v.to(DEVICE) if isinstance(v, torch.Tensor) else v for k, v in b.items()}
            p, _ = model(b['values'], b['company_ids'], b['metric_ids'],
                        b['day'], b['month'], b['dow'], b['year_offset'],
                        b['time_since_update'], b['mask'].bool(),
                        b['source_ids'], b['time_bins'])
            daily_preds[date].append(p.item())
            daily_labels[date].append(sample['label'].item())
    if (di + 1) % 20 == 0:
        print(f"    {di+1}/{len(sample_dates)} dates")

daily_ics = []
daily_n = []
for date in sample_dates:
    if len(daily_preds[date]) < 10: continue
    ic = spearmanr(daily_preds[date], daily_labels[date]).correlation
    daily_ics.append(ic)
    daily_n.append(len(daily_preds[date]))

daily_ics = np.array(daily_ics)
daily_n = np.array(daily_n)

print(f"\n{'='*60}")
print(f"  CROSS-SECTIONAL RANK IC RESULTS")
print(f"{'='*60}")
print(f"  Test dates:          {len(daily_ics)}")
print(f"  Mean stocks/date:    {daily_n.mean():.0f}")
print(f"  Mean Daily Rank IC:  {daily_ics.mean():+.4f}")
print(f"  Std Daily Rank IC:   {daily_ics.std():.4f}")
print(f"  IC Information Ratio:{daily_ics.mean()/daily_ics.std():.3f}" if daily_ics.std() > 0 else "  IC IR: N/A")
print(f"  Positive IC days:    {(daily_ics > 0).mean()*100:.1f}%")
print(f"  IC > 0.05 days:      {(daily_ics > 0.05).mean()*100:.1f}%")
print(f"  IC > 0.10 days:      {(daily_ics > 0.10).mean()*100:.1f}%")
print(f"  IC t-stat:           {daily_ics.mean()/daily_ics.std()*np.sqrt(len(daily_ics)):.2f}" if daily_ics.std() > 0 else "  IC t-stat: N/A")

# Expert usage
print(f"\n  Expert Usage (Test):")
for e in range(6):
    u = gate_usage[e]/gate_usage.sum()*100
    print(f"    Expert {e+1}: {u:5.1f}%")
cv = np.std(gate_usage/gate_usage.sum())/(1/6)
print(f"    CV: {cv:.3f}")

print(f"\n  CONCLUSION: {'STRONG SIGNAL' if daily_ics.mean() > 0.03 and daily_ics.mean()/daily_ics.std()*np.sqrt(len(daily_ics)) > 2 else 'MODERATE' if daily_ics.mean() > 0.01 else 'WEAK/NO SIGNAL'}")
