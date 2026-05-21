"""Check balance sheet files structure."""
import pandas as pd
import os, zipfile
from pathlib import Path

# Check the xlsx file
xlsx_path = Path("C:/Users/lcdell/Desktop/行业分类进度/资产负债表.xlsx")
if xlsx_path.exists():
    print(f"=== {xlsx_path.name} ({xlsx_path.stat().st_size/1e6:.1f} MB) ===")
    df = pd.read_excel(xlsx_path, nrows=5)
    print(f"Shape: {df.shape}, Cols: {list(df.columns)[:15]}")
    print(df.head(3).to_string(max_colwidth=30))

# Check ZIP files
zip_dir = Path("C:/Users/lcdell/Desktop/上市企业数据")
for zf in sorted(zip_dir.glob("*.zip")):
    print(f"\n=== {zf.name} ({zf.stat().st_size/1e6:.1f} MB) ===")
    with zipfile.ZipFile(zf) as z:
        names = z.namelist()
        print(f"  Contents: {names[:5]}")
        for name in names[:2]:
            if name.endswith(('.xlsx', '.xls', '.csv')):
                try:
                    df = pd.read_excel(z.open(name), nrows=3)
                    print(f"  {name}: {df.shape}, cols={list(df.columns)[:10]}")
                except:
                    print(f"  {name}: could not read")
