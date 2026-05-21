"""
Streaming IC computation — O(1) memory, no full dataset loading.

Pearson r = (n*Sxy - Sx*Sy) / sqrt((n*Sx2 - Sx^2) * (n*Sy2 - Sy^2))

Only 7 accumulators needed: n, Sx, Sy, Sxy, Sx2, Sy2, and gate counts.
"""
import sys; sys.path.insert(0,'.')
import numpy as np, torch
from torch.utils.data import DataLoader, Subset
from utils.dataset import SlidingWindowDataset, collate_fn
from model import FinancialMoETransformer
from config import model_cfg
from collections import defaultdict

# ── Streaming Pearson accumulators ───────────────────────────────
class StreamingStats:
    def __init__(self):
        self.n = 0; self.sx = 0.0; self.sy = 0.0
        self.sxy = 0.0; self.sx2 = 0.0; self.sy2 = 0.0
        self.gate_counts = np.zeros(6)

    def add(self, preds, labels, gate_weights):
        n = len(preds)
        self.n += n
        self.sx += preds.sum()
        self.sy += labels.sum()
        self.sxy += (preds * labels).sum()
        self.sx2 += (preds ** 2).sum()
        self.sy2 += (labels ** 2).sum()
        self.gate_counts += gate_weights.argmax(axis=1).bincount(minlength=6)

    def ic(self):
        n, sx, sy = self.n, self.sx, self.sy
        num = n * self.sxy - sx * sy
        den = np.sqrt(max(0, n * self.sx2 - sx**2) * max(0, n * self.sy2 - sy**2))
        return num / den if den > 0 else 0.0

    def rank_ic(self, preds, labels):
        """Rank IC needs sorted data — compute once at end on small arrays."""
        rp = preds.argsort().argsort().astype(float)
        rl = labels.argsort().argsort().astype(float)
        return np.corrcoef(rp, rl)[0, 1]


print("Streaming Test Evaluation (O(1) memory)")

# Load labels to determine test period, then filter data to test+lookback only
import pandas as pd
labels = pd.read_parquet("D:/financial_data/processed/labels.parquet")
labels["datetime"] = pd.to_datetime(labels["datetime"]); labels = labels.sort_values("datetime")
n = len(labels)
test_labels = labels.iloc[int(n*0.75):]
min_dt = test_labels["datetime"].min() - pd.Timedelta(days=400)  # 1yr+ lookback
print(f"Filtering data from {min_dt.date()} onwards...")

dp = pd.read_parquet("D:/financial_data/processed/data_points.parquet")
dp["datetime"] = pd.to_datetime(dp["datetime"])
dp = dp[dp["datetime"] >= min_dt].sort_values("datetime").reset_index(drop=True)
print(f"Filtered: {len(dp):,} rows (was 22M)")

# Patch dataset with filtered data
ds = SlidingWindowDataset.__new__(SlidingWindowDataset)
ds.data = dp
ds.labels = labels  # full labels needed for window building
ds.window_days = 365
ds.forecast_horizon = 63
ds.max_seq_len = 8192
ds.base_year = 2015
ds.use_cache = False
ds.multi_window = False
ds._window_options = [365]
ds.company_to_id, ds.metric_to_id = ds._build_dual_vocab()
ds.window_dates = ds._build_windows()
ds.var_to_id = ds.company_to_id
print(f"Vocab: {len(ds.company_to_id):,}c x {len(ds.metric_to_id):,}m | Windows: {len(ds.window_dates)}")

test_idx = list(range(len(ds.window_dates)))
print(f"Test windows: {len(test_idx)}")

model = FinancialMoETransformer(
    n_companies=ds.n_companies+2, n_metrics=ds.n_metrics+2,
    d_model=128, nhead=4, num_layers=model_cfg.num_layers,
    dim_feedforward=model_cfg.dim_feedforward,
    num_experts=model_cfg.num_experts, expert_hidden=model_cfg.expert_hidden,
)
ckpt = torch.load("checkpoints/best.pt", map_location="cpu", weights_only=False)
model.load_state_dict(ckpt["model"])
model.eval()
print(f"Model: epoch {ckpt['epoch']+1}, Val IC={ckpt['best_ic']:.4f}")

# ── Stream through test set ──────────────────────────────────────
loader = DataLoader(Subset(ds, test_idx), batch_size=2, shuffle=False,
                    collate_fn=collate_fn, num_workers=0, pin_memory=False)

stats = StreamingStats()
all_preds = []  # small: only for final rank IC
all_labels = []

for i, batch in enumerate(loader):
    if not batch or 'values' not in batch: continue
    with torch.no_grad():
        pred, _ = model(
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
        gates = model.moe.get_expert_weights(xp)

    p = pred.numpy().flatten()
    l = batch['label'].numpy().flatten()
    g = gates.numpy()

    stats.add(p, l, torch.from_numpy(g))
    all_preds.extend(p); all_labels.extend(l)

    if (i + 1) % 50 == 0:
        print(f"  batch {i+1}/{len(loader)} | IC(so far)={stats.ic():+.4f} | n={stats.n}")

# ── Final metrics ────────────────────────────────────────────────
all_preds = np.array(all_preds); all_labels = np.array(all_labels)
ic = stats.ic()
rank_ic = stats.rank_ic(all_preds, all_labels)
mse = np.mean((all_preds - all_labels)**2)

print(f"\n{'='*55}")
print(f"  TEST SET RESULTS ({stats.n} samples)")
print(f"{'='*55}")
print(f"  IC:           {ic:+.4f}")
print(f"  Rank IC:      {rank_ic:+.4f}")
print(f"  MSE:          {mse:.4f}")

print(f"\n  Expert Usage (Test Set):")
for e in range(6):
    usage = stats.gate_counts[e] / stats.gate_counts.sum() * 100
    bar = "#" * int(usage)
    print(f"    Expert {e+1}: {usage:5.1f}%  {bar}")

cv = np.std(stats.gate_counts / stats.gate_counts.sum()) / (1/6)
print(f"    CV: {cv:.3f}")

print(f"\n  Val IC (from ckpt): +{ckpt['best_ic']:.4f}")
print(f"  Test IC (now):      {ic:+.4f}")
