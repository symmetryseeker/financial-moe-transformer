"""Test eval - avoid loading full dataset."""
import sys; sys.path.insert(0,'.')
import numpy as np, pandas as pd, torch
from torch.utils.data import DataLoader, Subset
from utils.dataset import SlidingWindowDataset, collate_fn
from model import FinancialMoETransformer
from config import model_cfg

# Load labels only to get test date range
labels = pd.read_parquet("D:/financial_data/processed/labels.parquet")
labels["datetime"] = pd.to_datetime(labels["datetime"])
labels = labels.sort_values("datetime")

# Use only the last 500 windows for test
n_total = len(labels)
test_start = int(n_total * 0.75)
test_labels = labels.iloc[test_start:]

print(f"Test period: {test_labels['datetime'].min().date()} -> {test_labels['datetime'].max().date()} ({len(test_labels)} labels)")

# Get a small data subset for windows
dp = pd.read_parquet("D:/financial_data/processed/data_points.parquet")
dp["datetime"] = pd.to_datetime(dp["datetime"])

# Filter to test period + 1 year lookback
min_date = test_labels["datetime"].min() - pd.Timedelta(days=365)
dp = dp[dp["datetime"] >= min_date]
print(f"Filtered data: {len(dp):,} rows")

# Now load dataset with filtered data
# We'll override the data directly
ds = SlidingWindowDataset(max_seq_len=8192, use_cache=False)
# Override with filtered
ds.data = dp[dp["datetime"] >= min_date].sort_values("datetime").reset_index(drop=True)

# Rebuild windows and vocab
ds.company_to_id, ds.metric_to_id = ds._build_dual_vocab()
ds.window_dates = ds._build_windows()
print(f"Windows in test period: {len(ds.window_dates)}")

# Get model
model = FinancialMoETransformer(
    n_companies=ds.n_companies+2, n_metrics=ds.n_metrics+2,
    d_model=128, nhead=4, num_layers=model_cfg.num_layers,
    dim_feedforward=model_cfg.dim_feedforward,
    num_experts=model_cfg.num_experts, expert_hidden=model_cfg.expert_hidden,
)
ckpt = torch.load("checkpoints/best.pt", map_location="cpu", weights_only=False)
model.load_state_dict(ckpt["model"])
model.eval()
print(f"Loaded: epoch {ckpt['epoch']+1}, Val IC={ckpt['best_ic']:.4f}")

# Evaluate
loader = DataLoader(ds, batch_size=2, shuffle=False, collate_fn=collate_fn, num_workers=0)
P, L, G = [], [], []
for batch in loader:
    if not batch or 'values' not in batch: continue
    with torch.no_grad():
        p, _ = model(
            batch['values'], batch['company_ids'], batch['metric_ids'],
            batch['day'], batch['month'], batch['dow'],
            batch['year_offset'], batch['time_since_update'],
            batch['mask'].bool(), batch['source_ids'], batch['time_bins'],
        )
        x = model.embedding(
            batch['values'], batch['company_ids'], batch['metric_ids'],
            batch['day'], batch['month'], batch['dow'],
            batch['year_offset'], batch['time_since_update'],
        )
        x, hm = model.hierarchical(x, batch['mask'].bool(),
                                   batch['source_ids'], batch['time_bins'])
        xp = model.pooling(x, hm)
        G.append(model.moe.get_expert_weights(xp).numpy())
    P.extend(p.numpy().flatten())
    L.extend(batch['label'].numpy().flatten())

P = np.array(P); L = np.array(L); G = np.concatenate(G); T = G.argmax(1)
ic = np.corrcoef(P, L)[0, 1]
ric = np.corrcoef(P.argsort().argsort(), L.argsort().argsort())[0, 1]

print(f"\nTest IC: {ic:+.4f}  Rank IC: {ric:+.4f}  ({len(P)} samples)")
print(f"\nExpert Usage (Test Set):")
for e in range(6):
    u = (T == e).mean() * 100
    bar = "#" * int(u)
    print(f"  Expert {e+1}: {u:5.1f}%  {bar}")
cv = np.std([(T == e).mean() for e in range(6)]) / (1/6)
print(f"  CV: {cv:.3f}")
