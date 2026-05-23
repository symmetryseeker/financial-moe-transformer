"""
Build individual stock-level labels.
For each CSI 300 constituent stock, compute 63-day forward excess return
relative to the CSI 300 index.

Output: labels_stock.parquet with columns [datetime, stock_code, label]
"""
import sys; sys.path.insert(0, ".")
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import pyarrow.compute as pc
import pyarrow as pa
from pathlib import Path
# Use project-local data (same as training)
DATA_PATH = Path("data/processed/data_points.parquet")
LABELS_OUT = Path("data/processed/labels_stock.parquet")

FORECAST_HORIZON = 63
RF_ANNUAL = 0.025
daily_rf = (1 + RF_ANNUAL) ** (1/252) - 1

print("=" * 60)
print("  Building Stock-Level Labels")
print("=" * 60)

# ── Step 1: Extract all stock close prices + CSI 300 index ──
print("\n[1/3] Extracting stock close prices...")
pf = pq.ParquetFile(Path("data/processed") /"data_points.parquet")
stock_data = []  # (datetime, stock_code, close_price)
index_data = []  # (datetime, close_price)

for i in range(pf.metadata.num_row_groups):
    tbl = pf.read_row_group(i, columns=["datetime", "source", "variable", "value_raw"])
    df = tbl.to_pandas()
    # Individual stock closes: sh_600519::close format
    stock_mask = (
        (df["source"] == "market") &
        df["variable"].str.match(r"^(sh|sz)_\d+::close$", na=False)
    )
    if stock_mask.any():
        stock_df = df[stock_mask].copy()
        stock_df["stock_code"] = stock_df["variable"].str.split("::").str[0]
        stock_data.append(stock_df[["datetime", "stock_code", "value_raw"]])
    # CSI 300 index close: unqualified "close"
    index_mask = (df["source"] == "market") & (df["variable"] == "close") & (~df["value_raw"].isna())
    if index_mask.any():
        index_data.append(df[index_mask][["datetime", "value_raw"]])
    if (i + 1) % 5 == 0:
        print(f"  row group {i+1}/{pf.metadata.num_row_groups}")

stock_df = pd.concat(stock_data, ignore_index=True)
stock_df["datetime"] = pd.to_datetime(stock_df["datetime"])
stock_df["value_raw"] = pd.to_numeric(stock_df["value_raw"], errors="coerce")
stock_df = stock_df.dropna(subset=["value_raw"]).sort_values(["stock_code", "datetime"])

index_df = pd.concat(index_data, ignore_index=True)
index_df["datetime"] = pd.to_datetime(index_df["datetime"])
index_df["value_raw"] = pd.to_numeric(index_df["value_raw"], errors="coerce")
index_df = index_df.dropna(subset=["value_raw"]).drop_duplicates("datetime").sort_values("datetime")
index_df = index_df.rename(columns={"value_raw": "index_close"})

print(f"  Stocks: {stock_df['stock_code'].nunique()}, rows: {len(stock_df):,}")
print(f"  Index: {len(index_df):,} trading days")

# ── Step 2: Compute forward returns ──
print("\n[2/3] Computing forward excess returns...")
labels_list = []

for stock, grp in stock_df.groupby("stock_code"):
    grp = grp.sort_values("datetime").copy()
    # Forward return: ln(P_{t+63} / P_t)
    grp["fwd_close"] = grp["value_raw"].shift(-FORECAST_HORIZON)
    grp["stock_ret"] = np.log(grp["fwd_close"] / grp["value_raw"].replace(0, np.nan))
    grp["stock_ret"] = grp["stock_ret"].replace([np.inf, -np.inf], np.nan)

    # Merge with index
    grp = grp.merge(index_df, on="datetime", how="left")
    grp = grp.sort_values("datetime")
    grp["fwd_index"] = grp["index_close"].shift(-FORECAST_HORIZON)
    grp["index_ret"] = np.log(grp["fwd_index"] / grp["index_close"].replace(0, np.nan))
    grp["index_ret"] = grp["index_ret"].replace([np.inf, -np.inf], np.nan)

    # Excess return: stock - index, subtract risk-free
    grp["label"] = grp["stock_ret"] - grp["index_ret"] - daily_rf * FORECAST_HORIZON

    # Volatility normalization: divide by rolling 252-day std of stock excess returns
    grp["label_vol"] = grp["label"].rolling(252, min_periods=63).std()
    grp["label"] = grp["label"] / grp["label_vol"].replace(0, np.nan)

    valid = grp.dropna(subset=["label"])[["datetime", "stock_code", "label"]]
    if len(valid) > 0:
        labels_list.append(valid)

    if len(labels_list) % 50 == 0:
        print(f"  {len(labels_list)}/{stock_df['stock_code'].nunique()} stocks")

all_labels = pd.concat(labels_list, ignore_index=True)
all_labels = all_labels.sort_values(["datetime", "stock_code"]).reset_index(drop=True)

# ── Step 3: Save ──
print(f"\n[3/3] Saving...")
out_path = Path("data/processed") /"labels_stock.parquet"
all_labels.to_parquet(out_path, index=False)

print(f"\n{'=' * 60}")
print(f"  Stock labels saved to {out_path}")
print(f"  Total labels: {len(all_labels):,}")
print(f"  Unique stocks: {all_labels['stock_code'].nunique()}")
print(f"  Date range: {all_labels['datetime'].min().date()} -> {all_labels['datetime'].max().date()}")
print(f"  Dates: {all_labels['datetime'].nunique()}")
print(f"  Labels per date: {len(all_labels) / all_labels['datetime'].nunique():.0f}")
print(f"  Label stats: mean={all_labels['label'].mean():.4f} std={all_labels['label'].std():.4f}")
print(f"  Pos/Neg: {(all_labels['label']>0).mean()*100:.1f}% / {(all_labels['label']<0).mean()*100:.1f}%")
print(f"{'=' * 60}")
