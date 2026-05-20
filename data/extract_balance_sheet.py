"""
Extract balance sheet + cashflow statement from Desktop ZIP files
into D:/financial_data/financial/ for the data pipeline.
"""
import sys, zipfile, io, os
from pathlib import Path
import pandas as pd
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

ZIP_DIR = Path("C:/Users/lcdell/Desktop/上市企业数据")
OUT_DIR = Path("D:/financial_data/financial")
OUT_DIR.mkdir(parents=True, exist_ok=True)

def extract_and_save(zip_path: Path, output_name: str):
    """Extract the largest XLSX from a ZIP and save as CSV."""
    if not zip_path.exists():
        print(f"  NOT FOUND: {zip_path}")
        return

    with zipfile.ZipFile(zip_path) as z:
        # Find data files (CSV or XLSX)
        data_files = [n for n in z.namelist()
                      if n.endswith(('.csv', '.xlsx', '.xls', '.txt'))
                      and not n.startswith(('版权', 'license', 'readme'))]
        if not data_files:
            print(f"  No data files in {zip_path.name}")
            return

        # Use the largest one
        sizes = [(n, z.getinfo(n).file_size) for n in data_files]
        sizes.sort(key=lambda x: x[1], reverse=True)
        main_file = sizes[0][0]

        print(f"  Extracting: {main_file} ({sizes[0][1]/1e6:.1f} MB)")

        with z.open(main_file) as f:
            suffix = Path(main_file).suffix.lower()
            if suffix in ('.xlsx', '.xls'):
                df = pd.read_excel(io.BytesIO(f.read()))
            else:
                # CSV — use streaming to avoid memory issues
                # Read header first to get column names
                header_line = f.readline().decode('utf-8', errors='replace')
                # Try to detect encoding by reading a sample
                f.seek(0)
                raw_sample = f.read(10000)
                for enc in ['utf-8', 'gbk', 'gb18030', 'gb2312']:
                    try:
                        raw_sample.decode(enc)
                        detected_enc = enc
                        break
                    except UnicodeDecodeError:
                        continue
                else:
                    detected_enc = 'gb18030'

                f.seek(0)
                # Read in chunks
                chunks = []
                for chunk in pd.read_csv(f, encoding=detected_enc,
                                         chunksize=50000, low_memory=False):
                    chunks.append(chunk)
                df = pd.concat(chunks, ignore_index=True)
                print(f"    Read {len(chunks)} chunks")

        print(f"    Shape: {df.shape}, Cols: {len(df.columns)}")

        # Save
        out_path = OUT_DIR / f"{output_name}.csv"
        df.to_csv(out_path, index=False)
        print(f"    Saved: {out_path} ({len(df):,} rows)")

# Main tables
targets = [
    ("资产负债表153956739(仅供北京理工大学使用).zip", "balance_sheet"),
    ("利润表153812588(仅供北京理工大学使用).zip", "income_statement_full"),
    # Cashflow files exist by sector
]

print("=" * 60)
print("  EXTRACTING FINANCIAL STATEMENTS")
print("=" * 60)

for zip_name, out_name in targets:
    extract_and_save(ZIP_DIR / zip_name, out_name)

# Also extract sector-specific balance sheets
for sector, prefix in [("银行", "bank"), ("证券", "securities"), ("保险", "insurance")]:
    for f in ZIP_DIR.glob(f"{sector}类资产负债表*.zip"):
        extract_and_save(f, f"balance_sheet_{prefix}")

# Cashflow statements
for sector, prefix in [("银行", "bank"), ("证券", "securities"), ("保险", "insurance")]:
    for f in ZIP_DIR.glob(f"{sector}类现金流量表*.zip"):
        extract_and_save(f, f"cashflow_{prefix}")

print(f"\nDone. Files in {OUT_DIR}:")
for f in sorted(OUT_DIR.glob("*.csv")):
    print(f"  {f.name} ({f.stat().st_size/1e6:.1f} MB)")
