import requests, json, time, sys
from pathlib import Path
import pandas as pd

DATA_DIR = Path("D:/financial_data")

tickers = {
    "^VIX": ("macro", "vix_daily"),
    "DX-Y.NYB": ("macro", "dxy_daily"),
    "^TNX": ("macro", "us10y_daily"),
    "^GSPC": ("macro", "sp500_daily"),
    "^HSI": ("macro", "hsi_daily"),
}

ok = 0
for symbol, (cat, name) in tickers.items():
    url = f"https://query1.finance.yahoo.com/v8/finance/chart/{symbol}?range=10y&interval=1d"
    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}
    try:
        r = requests.get(url, headers=headers, timeout=15)
        if r.status_code == 200:
            data = r.json()
            result = data["chart"]["result"][0]
            ts = result["timestamp"]
            quotes = result["indicators"]["quote"][0]
            df = pd.DataFrame({
                "date": pd.to_datetime(ts, unit="s"),
                "open": quotes["open"],
                "high": quotes["high"],
                "low": quotes["low"],
                "close": quotes["close"],
                "volume": quotes["volume"],
            })
            df = df.dropna(subset=["close"])
            outdir = DATA_DIR / cat
            outdir.mkdir(parents=True, exist_ok=True)
            df.to_csv(outdir / f"{name}.csv", index=False)
            print(f"  {name}: OK ({len(df)} rows)")
            ok += 1
        else:
            print(f"  {name}: HTTP {r.status_code} — {r.text[:100]}")
        time.sleep(3)
    except Exception as e:
        print(f"  {name}: {str(e)[:80]}")

print(f"\nDirect API: {ok}/{len(tickers)} OK")
