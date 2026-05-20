"""
Lightweight label + volatility feature rebuild.
Reads existing data_points.parquet in streaming fashion.
Memory: <500MB. Time: ~2 min.
"""
import sys
sys.path.insert(0, ".")
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import pyarrow as pa
import pyarrow.compute as pc
from pathlib import Path
from config import DATA_PROCESSED

FORECAST_HORIZON = 63
RF_ANNUAL = 0.025

print("=" * 60)
print("  Lightweight Label & Volatility Feature Rebuild")
print("=" * 60)

# ── Step 1: Extract CSI 300 close prices (streaming) ──
print("\n[1/4] Extracting CSI 300 close prices...")
pf = pq.ParquetFile(DATA_PROCESSED / "data_points.parquet")
csi_rows = []

# Only read rows that match CSI 300 index close (unqualified "close" = index-level)
for i in range(pf.metadata.num_row_groups):
    tbl = pf.read_row_group(i, columns=["datetime", "source", "variable", "value_raw"])
    df_chunk = tbl.to_pandas()
    mask = (
        (df_chunk["source"] == "market") &
        (df_chunk["variable"] == "close") &
        (~df_chunk["value_raw"].isna())
    )
    if mask.any():
        csi_rows.append(df_chunk[mask][["datetime", "value_raw"]])
    if (i + 1) % 5 == 0:
        print(f"  row group {i+1}/{pf.metadata.num_row_groups}")

csi300 = pd.concat(csi_rows, ignore_index=True)
csi300 = csi300.drop_duplicates(subset=["datetime"]).sort_values("datetime")
csi300["value_raw"] = pd.to_numeric(csi300["value_raw"], errors="coerce")
csi300 = csi300.dropna(subset=["value_raw"])
print(f"  CSI 300: {len(csi300)} trading days, {csi300['datetime'].min().date()} -> {csi300['datetime'].max().date()}")

# ── Step 2: Compute labels + volatility features ──
print("\n[2/4] Computing labels and volatility features...")

# Daily log returns
csi300["ret"] = np.log(csi300["value_raw"] / csi300["value_raw"].shift(1))
csi300["ret"] = csi300["ret"].replace([np.inf, -np.inf], np.nan)

# Realized volatility (annualized) — from PAST returns only
for w in [21, 63, 252]:
    csi300[f"vol_{w}d"] = csi300["ret"].rolling(w, min_periods=max(5, w//4)).std() * np.sqrt(252)

# Build volatility feature rows
vol_rows = []
for w in [21, 63, 252]:
    col = f"vol_{w}d"
    sub = csi300[["datetime", col]].dropna().copy()
    sub = sub.rename(columns={col: "value_raw"})
    sub["source"] = "macro"
    sub["variable"] = f"csi300_realized_vol_{w}d"
    sub["value"] = sub["value_raw"]
    sub["time_since_update"] = 0.0
    vol_rows.append(sub[["datetime", "source", "variable", "value", "value_raw", "time_since_update"]])
vol_df = pd.concat(vol_rows, ignore_index=True)

# Expanding z-score per volatility variable
for v in vol_df["variable"].unique():
    mask = vol_df["variable"] == v
    vals = vol_df.loc[mask, "value_raw"].values
    em = pd.Series(vals).expanding(min_periods=63).mean().values
    es = pd.Series(vals).expanding(min_periods=63).std().values
    vol_df.loc[mask, "value"] = (vals - em) / np.where(es == 0, 1, es)
vol_df = vol_df.dropna(subset=["value"])
print(f"  Volatility features: {len(vol_df):,} rows (21d/63d/252d, expanding z-scored)")

# Label: raw 63-day excess log-return, winsorized at 1%/99%
csi300["fwd"] = csi300["value_raw"].shift(-FORECAST_HORIZON)
ratio = csi300["fwd"] / csi300["value_raw"].replace(0, np.nan)
csi300["label"] = np.log(ratio.astype(float))
daily_rf = (1 + RF_ANNUAL) ** (1/252) - 1
csi300["label"] = csi300["label"] - daily_rf * FORECAST_HORIZON
# Volatility normalization — proven formula, keeps label std≈1
csi300["label_vol"] = csi300["label"].rolling(252, min_periods=63).std()
csi300["label"] = csi300["label"] / csi300["label_vol"].replace(0, np.nan)
labels = csi300.dropna(subset=["label"])[["datetime", "label"]].copy()
print(f"  Labels: {len(labels):,} rows")
print(f"  Label stats: mean={labels['label'].mean():.4f} std={labels['label'].std():.4f}")
print(f"  Pos/Neg: {(labels['label']>0).mean()*100:.1f}% / {(labels['label']<0).mean()*100:.1f}%")

# ── Step 3: Save labels ──
print(f"\n[3/4] Saving labels to {DATA_PROCESSED / 'labels.parquet'}...")
labels.to_parquet(DATA_PROCESSED / "labels.parquet", index=False)

# ── Step 4: Append volatility rows to data_points (streaming write) ──
print(f"[4/4] Appending volatility features to data_points.parquet...")

import tempfile, shutil

dp_path = DATA_PROCESSED / "data_points.parquet"
tmp_path = DATA_PROCESSED / "data_points_tmp.parquet"

# Open reader for existing file
reader = pq.ParquetFile(dp_path)
schema = reader.schema_arrow

# Convert vol_df to pyarrow Table (match schema)
vol_df["value"] = pd.to_numeric(vol_df["value"], errors="coerce")
vol_df["value_raw"] = pd.to_numeric(vol_df["value_raw"], errors="coerce")
vol_df["time_since_update"] = pd.to_numeric(vol_df["time_since_update"], errors="coerce").fillna(0)
vol_table = pa.Table.from_pandas(vol_df, schema=schema)

# Stream: read row groups, filter old vol rows, write to temp file
writer = pq.ParquetWriter(tmp_path, schema)
total_before = 0
total_after = 0

for i in range(reader.metadata.num_row_groups):
    tbl = reader.read_row_group(i, columns=["variable"])
    vars_arr = tbl.column("variable")
    # Filter: drop old csi300_realized_vol rows
    is_vol = pc.match_substring(vars_arr, "csi300_realized_vol")
    keep_mask = pc.invert(pc.fill_null(is_vol, False))

    # Read full row group, apply filter
    full_tbl = reader.read_row_group(i)
    filtered_tbl = full_tbl.filter(keep_mask)

    total_before += full_tbl.num_rows
    total_after += filtered_tbl.num_rows
    writer.write_table(filtered_tbl)

    if (i + 1) % 5 == 0:
        print(f"  row group {i+1}/{reader.metadata.num_row_groups}")

removed = total_before - total_after
print(f"  Existing: {total_before:,} rows (removed {removed:,} old vol rows)")

# Append new volatility rows
writer.write_table(vol_table)
writer.close()

print(f"  New vol features: {vol_table.num_rows:,} rows")
print(f"  Total: {total_after + vol_table.num_rows:,} rows")

# Atomic replace
shutil.move(str(tmp_path), str(dp_path))

print(f"\n{'='*60}")
print(f"  Done!")
print(f"  data_points.parquet: {total_after + vol_table.num_rows:,} rows")
print(f"  labels.parquet:      {len(labels):,} rows")
print(f"  New label: mean={labels['label'].mean():.4f} std={labels['label'].std():.4f}")
print(f"{'='*60}")
