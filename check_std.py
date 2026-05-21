import pandas as pd, numpy as np
# Read just the value column to save memory
dp = pd.read_parquet("D:/financial_data/processed/data_points.parquet", columns=["value","source"])
vals = dp["value"].dropna()
# Sample 1M rows for quantile computation
sample = vals.sample(min(1000000, len(vals)), random_state=42)
print(f"Value std after winsorize: {vals.std():.4f} (was ~1.27, target ~1.0)")
print(f"P1: {sample.quantile(0.01):.3f}  P99: {sample.quantile(0.99):.3f}")
print(f"Mean: {vals.mean():.4f}")
for src in sorted(dp["source"].unique()):
    s = dp[dp["source"]==src]["value"].dropna()
    print(f"  {src}: std={s.std():.4f}")

