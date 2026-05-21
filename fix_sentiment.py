"""Diagnose and fix sentiment/northbound flow data."""
import pandas as pd
from pathlib import Path

DATA_DIR = Path("D:/financial_data")

print("=" * 60)
print("  1. DIAGNOSE: Northbound flow date format")
print("=" * 60)

nb_path = DATA_DIR / "sentiment" / "northbound_flow.csv"
if nb_path.exists():
    df = pd.read_csv(nb_path)
    print(f"  Rows: {len(df)}, Cols: {list(df.columns)}")
    print(f"  Dtypes:\n{df.dtypes}")
    print(f"  Head (3 rows):")
    print(df.head(3).to_string())
    print(f"  trade_date sample: {df['trade_date'].iloc[:5].tolist()}")

    # Try parsing
    trade_dates = pd.to_datetime(df['trade_date'].astype(str), format='%Y%m%d', errors='coerce')
    print(f"  Parsed as YYYYMMDD: {trade_dates.notna().sum()}/{len(trade_dates)} OK")
    print(f"  Range: {trade_dates.min()} -> {trade_dates.max()}")
else:
    print("  FILE NOT FOUND")

print()
print("=" * 60)
print("  2. DIAGNOSE: All sentiment CSVs in D:/financial_data/")
print("=" * 60)
for f in DATA_DIR.rglob("*.csv"):
    if "sentiment" in str(f) or "northbound" in str(f) or "margin" in str(f):
        try:
            df = pd.read_csv(f, nrows=3)
            print(f"\n  {f.relative_to(DATA_DIR)}: {len(df)} rows (preview), cols={list(df.columns)[:8]}")
            # Check for date-like columns
            for c in df.columns:
                sample = str(df[c].iloc[0]) if len(df) > 0 else ""
                if any(kw in c.lower() for kw in ['date', 'time', 'trade', '日期']):
                    print(f"    [{c}]: {sample}")
        except Exception as e:
            print(f"  {f.relative_to(DATA_DIR)}: ERROR {e}")

print()
print("=" * 60)
print("  3. FIX: Re-parse and save corrected northbound data")
print("=" * 60)

if nb_path.exists():
    df = pd.read_csv(nb_path)
    # Parse date
    df['trade_date'] = pd.to_datetime(df['trade_date'].astype(str), format='%Y%m%d', errors='coerce')
    df = df.rename(columns={'trade_date': 'date'})
    df = df.dropna(subset=['date'])

    if 'date' in df.columns:
        print(f"  Fixed: {len(df)} rows, date range: {df['date'].min()} -> {df['date'].max()}")
        # Save fixed version
        out_path = DATA_DIR / "sentiment" / "northbound_flow_fixed.csv"
        df.to_csv(out_path, index=False)
        print(f"  Saved: {out_path}")

        # Show sample
        print(f"  Sample:")
        print(df[['date', 'north_money']].head(5).to_string() if 'north_money' in df.columns else df.head(3).to_string())
