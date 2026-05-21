"""Comprehensive data quality analysis for the technical review."""
import sys; sys.path.insert(0, '.')
import pandas as pd
import numpy as np

print("=" * 70)
print("  DATA QUALITY DEEP-DIVE")
print("=" * 70)

# 1. Data points overview
dp = pd.read_parquet("data/processed/data_points.parquet")
print(f"\n>>> DATA POINTS: {len(dp):,} rows")

# Per source stats
for src in sorted(dp["source"].unique()):
    sdf = dp[dp["source"] == src]
    n_vars = sdf["variable"].nunique()
    date_min = sdf["datetime"].min()
    date_max = sdf["datetime"].max()
    n_dates = sdf["datetime"].nunique()
    mean_val = sdf["value"].mean()
    std_val = sdf["value"].std()
    miss_pct = sdf["value"].isna().mean() * 100
    print(f"\n  {src}: {len(sdf):,} rows, {n_vars} vars, {n_dates} unique dates")
    print(f"    Date range: {date_min} -> {date_max}")
    print(f"    Value: mean={mean_val:.2f}, std={std_val:.2f}, missing={miss_pct:.1f}%")
    # Show top variables
    top_vars = sdf.groupby("variable").size().sort_values(ascending=False).head(5)
    print(f"    Top vars: {dict(top_vars)}")

# 2. Labels
labels = pd.read_parquet("data/processed/labels.parquet")
print(f"\n\n>>> LABELS: {len(labels):,} rows")
print(f"    Date range: {labels['datetime'].min()} -> {labels['datetime'].max()}")
print(f"    Mean: {labels['label'].mean():.4f}, Std: {labels['label'].std():.4f}")
print(f"    Skew: {labels['label'].skew():.2f}, Kurtosis: {labels['label'].kurtosis():.2f}")
print(f"    Min: {labels['label'].min():.4f}, Max: {labels['label'].max():.4f}")
print(f"    |label| > 0.05: {(abs(labels['label']) > 0.05).mean()*100:.1f}%")
print(f"    |label| > 0.10: {(abs(labels['label']) > 0.10).mean()*100:.1f}%")

# 3. Sequence length distribution
from utils.dataset import SlidingWindowDataset
ds = SlidingWindowDataset(
    data_path='data/processed/data_points.parquet',
    labels_path='data/processed/labels.parquet',
    window_days=365, forecast_horizon=21, max_seq_len=2048, use_cache=False
)
seq_lens = []
for i in range(min(500, len(ds))):
    s = ds[i]
    if s is not None:
        seq_lens.append(s["seq_len"])

seq_lens = np.array(seq_lens)
print(f"\n\n>>> SEQUENCE LENGTH DISTRIBUTION (n={len(seq_lens)})")
print(f"    Mean: {seq_lens.mean():.0f}, Median: {np.median(seq_lens):.0f}")
print(f"    Min: {seq_lens.min()}, Max: {seq_lens.max()}")
for pct in [50, 75, 90, 95, 99]:
    print(f"    P{pct}: {np.percentile(seq_lens, pct):.0f}")

# 4. Variable frequency analysis
print(f"\n\n>>> VARIABLE FREQUENCY (by source)")
for src in sorted(dp["source"].unique()):
    sdf = dp[dp["source"] == src]
    # For each variable, check average interval between updates
    for var in sdf["variable"].unique()[:3]:
        vdf = sdf[sdf["variable"] == var].sort_values("datetime")
        if len(vdf) > 2:
            intervals = vdf["datetime"].diff().dropna()
            avg_interval = intervals.mean()
            print(f"    {src}/{var}: {len(vdf):,} obs, avg interval={avg_interval}")

# 5. Time_since_update stats
print(f"\n\n>>> TIME_SINCE_UPDATE STATS")
tsu = dp["time_since_update"].dropna()
for pct in [50, 75, 90, 95, 99]:
    print(f"    P{pct}: {np.percentile(tsu, pct):.0f} days")
