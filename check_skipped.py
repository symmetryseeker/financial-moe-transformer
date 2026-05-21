"""Check why some files are being skipped."""
import pandas as pd
from pathlib import Path

files_to_check = [
    "D:/financial_data/macro/money_supply_ak.csv",
    "D:/financial_data/macro/pmi_ak.csv",
    "D:/financial_data/macro/money_supply.csv",
    "D:/financial_data/macro/money_supply_bs.csv",
]

for f in files_to_check:
    p = Path(f)
    if not p.exists():
        print(f"\n{p.name}: NOT FOUND")
        continue
    print(f"\n{'='*40}")
    print(f"  {p.name}")
    print(f"{'='*40}")
    df = pd.read_csv(f, nrows=5)
    print(f"  Shape: {df.shape}")
    print(f"  Columns: {list(df.columns)[:10]}")
    print(f"  Dtypes:")
    for c in df.columns[:8]:
        print(f"    {c}: {df[c].dtype}  sample={df[c].iloc[0]}")
    # Check for date-like
    for c in df.columns:
        if any(kw in str(c) for kw in ['月', '年', 'year', 'date', 'time', 'stat']):
            vals = df[c].head(5).tolist()
            print(f"  Date candidate [{c}]: {vals}")
    # Check for numeric issues
    for c in df.columns:
        sample = str(df[c].iloc[0])[:60]
        if any(ch in sample for ch in ['%', '亿', '万', ',', '，']):
            print(f"  Needs cleaning [{c}]: {sample}")
