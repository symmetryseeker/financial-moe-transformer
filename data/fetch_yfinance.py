"""
Fetch international macro via yfinance — with rate-limit handling.
"""
import sys, time
from pathlib import Path
import pandas as pd
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))
DATA_DIR = Path("D:/financial_data")

import yfinance as yf

def fetch_one(symbol, name, category, start="2015-01-01", max_retries=5):
    """Fetch a single ticker with retry + backoff."""
    for attempt in range(max_retries):
        try:
            ticker = yf.download(symbol, start=start, progress=False)
            if ticker is not None and not ticker.empty:
                df = ticker.reset_index()
                if isinstance(df.columns, pd.MultiIndex):
                    df.columns = [f"{c[0]}_{c[1]}".lower().strip('_') for c in df.columns]
                else:
                    df.columns = [str(c).lower() for c in df.columns]
                df = df.rename(columns={"date": "date", "Date": "date"})
                outdir = DATA_DIR / category
                outdir.mkdir(parents=True, exist_ok=True)
                outpath = outdir / f"{name}.csv"
                df.to_csv(outpath, index=False)
                print(f"  {name}: OK ({len(df)} rows)")
                return True

            print(f"  {name}: attempt {attempt+1} empty, retrying...")
            time.sleep(5 * (attempt + 1))

        except Exception as e:
            msg = str(e)[:80]
            if "Rate limit" in msg or "Too Many" in msg:
                wait = 10 * (attempt + 1)
                print(f"  {name}: rate-limited, waiting {wait}s...")
                time.sleep(wait)
            else:
                print(f"  {name}: {msg}")
                time.sleep(5)

    print(f"  {name}: FAILED after {max_retries} attempts")
    return False

print("=" * 60)
print("  YFINANCE INTERNATIONAL MACRO (with retry)")
print("=" * 60)

# Core tickers with 30s delay between them
tickers = [
    ("^VIX",       "vix_daily",             "macro"),
    ("DX-Y.NYB",   "dxy_daily",             "macro"),
    ("^TNX",       "us10y_daily",           "macro"),
    ("^GSPC",      "sp500_daily",           "macro"),
    ("^HSI",       "hsi_daily",             "macro"),
    ("CNY=X",      "usdcny_daily",          "macro"),
    ("GC=F",       "gold_comex_daily",      "alternative"),
    ("CL=F",       "crude_oil_wti_daily",   "alternative"),
]

ok = 0
for i, (symbol, name, cat) in enumerate(tickers):
    print(f"\n[{i+1}/{len(tickers)}] {symbol} ({name})")
    if fetch_one(symbol, name, cat):
        ok += 1
    # Long delay between tickers to avoid rate limiting
    if i < len(tickers) - 1:
        time.sleep(8)

# Summary
files = list(DATA_DIR.rglob("*.csv"))
size = sum(f.stat().st_size for f in files) / 1e6
print(f"\n{'='*60}")
print(f"  YFINANCE: {ok}/{len(tickers)} OK")
print(f"  Total: {len(files)} files, {size:.1f} MB")
