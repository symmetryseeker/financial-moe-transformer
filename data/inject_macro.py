"""
Inject international macro CSVs into existing data_points.parquet.
Reads CSV files from D:/financial_data/macro/ and alternative/,
standardizes to long format, z-scores, and appends.
"""
import sys; sys.path.insert(0, ".")
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import pyarrow as pa
import pyarrow.compute as pc
from pathlib import Path
import shutil

DATA_DIR = Path("D:/financial_data")
DP_PATH = Path("data/processed/data_points.parquet")
TMP_PATH = Path("data/processed/data_points_tmp.parquet")

# Files to inject (source -> variable name, CSV path)
MACRO_FILES = {
    "macro": {
        "vix": "macro/vix_daily.csv",
        "dxy": "macro/dxy_daily.csv",
        "us10y": "macro/us10y_daily.csv",
        "sp500": "macro/sp500_daily.csv",
        "hsi": "macro/hsi_daily.csv",
        "nikkei225": "macro/nikkei225_daily.csv",
        "stoxx50": "macro/stoxx50_daily.csv",
        "eem": "macro/eem_daily.csv",
        "fxi": "macro/fxi_daily.csv",
    },
    "alternative": {
        "gold": "alternative/gold_comex_daily.csv",
        "crude_oil": "alternative/crude_oil_wti_daily.csv",
        "silver": "alternative/silver_comex_daily.csv",
        "copper": "alternative/copper_comex_daily.csv",
    },
}

def load_csv(path):
    """Load a CSV, detect date column, return (dates, values)."""
    df = pd.read_csv(DATA_DIR / path)
    # Find date column
    date_col = None
    for c in df.columns:
        if c.lower() in ("date", "datetime", "time", "timestamp", "trade_date"):
            date_col = c
            break
    if date_col is None:
        # Try first column
        date_col = df.columns[0]
    dates = pd.to_datetime(df[date_col], errors="coerce")
    # Find value column (last numeric column)
    val_col = None
    for c in reversed(df.columns):
        if c != date_col and df[c].dtype in (np.float64, np.int64):
            val_col = c
            break
    if val_col is None:
        for c in reversed(df.columns):
            if c != date_col:
                try:
                    df[c] = pd.to_numeric(df[c], errors="coerce")
                    val_col = c
                    break
                except:
                    pass
    if val_col is None:
        return None
    vals = pd.to_numeric(df[val_col], errors="coerce")
    return pd.DataFrame({"datetime": dates, "value_raw": vals}).dropna()

print("=== Injecting International Macro Data ===\n")

new_rows = []
total_added = 0
for source, files in MACRO_FILES.items():
    for vname, csv_path in files.items():
        full_path = DATA_DIR / csv_path
        if not full_path.exists():
            print(f"  SKIP {csv_path}: not found")
            continue
        data = load_csv(csv_path)
        if data is None or len(data) == 0:
            print(f"  SKIP {csv_path}: no data")
            continue
        data["source"] = source
        data["variable"] = vname
        data["time_since_update"] = 0.0

        # Expanding z-score (forward-looking safe)
        data = data.sort_values("datetime")
        raw = data["value_raw"].values
        em = pd.Series(raw).expanding(min_periods=63).mean().values
        es = pd.Series(raw).expanding(min_periods=63).std().values
        data["value"] = (raw - em) / np.where(es == 0, 1, es)
        data = data.dropna(subset=["value"])

        new_rows.append(data[["datetime", "source", "variable", "value", "value_raw", "time_since_update"]])
        total_added += len(data)
        print(f"  [{source}] {vname}: {len(data)} rows, {data['datetime'].min().date()} -> {data['datetime'].max().date()}")

new_df = pd.concat(new_rows, ignore_index=True)
print(f"\nTotal new rows: {len(new_df):,}")

# Streaming append to data_points.parquet
print(f"\nAppending to {DP_PATH}...")

reader = pq.ParquetFile(DP_PATH)
schema = reader.schema_arrow
# Reorder new_df columns to match existing schema
col_order = [f.name for f in schema]
new_df = new_df[col_order]
new_table = pa.Table.from_pandas(new_df, schema=schema)

with pq.ParquetWriter(TMP_PATH, schema) as writer:
    for i in range(reader.metadata.num_row_groups):
        tbl = reader.read_row_group(i)
        writer.write_table(tbl)
        if (i + 1) % 5 == 0:
            print(f"  copied RG {i+1}/{reader.metadata.num_row_groups}")
    writer.write_table(new_table)

old_rows = reader.metadata.num_rows
new_total = old_rows + len(new_df)
shutil.move(str(TMP_PATH), str(DP_PATH))
print(f"\nDone: {old_rows:,} -> {new_total:,} rows (+{len(new_df):,})")
print(f"International macro data injected successfully.")
