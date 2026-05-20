"""
Fetch international macro via direct Yahoo Finance API (bypasses yfinance).
"""
import requests, json, time
from pathlib import Path
import pandas as pd

DATA_DIR = Path("D:/financial_data")

def fetch_yahoo(symbol, name, category, range_str="10y"):
    """Fetch chart data directly from Yahoo Finance v8 API."""
    url = f"https://query1.finance.yahoo.com/v8/finance/chart/{symbol}?range={range_str}&interval=1d"
    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}
    try:
        r = requests.get(url, headers=headers, timeout=15)
        if r.status_code != 200:
            print(f"    HTTP {r.status_code}")
            return False
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
        if df.empty:
            return False
        outdir = DATA_DIR / category
        outdir.mkdir(parents=True, exist_ok=True)
        df.to_csv(outdir / f"{name}.csv", index=False)
        print(f"    -> {name}.csv ({len(df)} rows, {df['date'].min().date()} → {df['date'].max().date()})")
        return True
    except Exception as e:
        print(f"    {str(e)[:80]}")
        return False

print("=" * 60)
print("  YAHOO FINANCE DIRECT API")
print("=" * 60)

ok = 0
tickers = [
    # Volatility & Sentiment
    ("^VIX",       "vix_daily",             "macro",        "CBOE Volatility Index"),
    # USD & Rates
    ("DX-Y.NYB",   "dxy_daily",             "macro",        "US Dollar Index"),
    ("^TNX",       "us10y_daily",           "macro",        "US 10Y Treasury Yield"),
    ("^IRX",       "us3m_daily",            "macro",        "US 3M Treasury Bill"),
    # Equity indices
    ("^GSPC",      "sp500_daily",           "macro",        "S&P 500"),
    ("^HSI",       "hsi_daily",             "macro",        "Hang Seng Index"),
    ("^N225",      "nikkei225_daily",       "macro",        "Nikkei 225"),
    ("^FTSE",      "ftse100_daily",         "macro",        "FTSE 100"),
    ("^STOXX50E",  "stoxx50_daily",         "macro",        "Euro Stoxx 50"),
    # Commodities
    ("GC=F",       "gold_comex_daily",      "alternative",  "Gold COMEX"),
    ("CL=F",       "crude_oil_wti_daily",   "alternative",  "Crude Oil WTI"),
    ("SI=F",       "silver_comex_daily",    "alternative",  "Silver COMEX"),
    ("HG=F",       "copper_comex_daily",    "alternative",  "Copper COMEX"),
    # FX
    ("CNY=X",      "usdcny_daily",          "macro",        "USD/CNY"),
    ("EURUSD=X",   "eurusd_daily",          "macro",        "EUR/USD"),
    ("JPY=X",      "usdjpy_daily",          "macro",        "USD/JPY"),
    # China-related (cross-check)
    ("FXI",        "fxi_daily",             "macro",        "iShares China Large-Cap ETF"),
    ("MCHI",       "mchi_daily",            "macro",        "iShares MSCI China ETF"),
    # Emerging markets
    ("EEM",        "eem_daily",             "macro",        "iShares MSCI Emerging Markets"),
]

for i, (symbol, name, cat, desc) in enumerate(tickers):
    print(f"[{i+1:2d}/{len(tickers)}] {desc} ({symbol})")
    if fetch_yahoo(symbol, name, cat):
        ok += 1
    time.sleep(2)  # polite delay

files = list(DATA_DIR.rglob("*.csv"))
size = sum(f.stat().st_size for f in files) / 1e6
print(f"\n{'='*60}")
print(f"  YAHOO DIRECT: {ok}/{len(tickers)} OK")
print(f"  Total: {len(files)} files, {size:.1f} MB")
