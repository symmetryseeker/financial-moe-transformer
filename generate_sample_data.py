"""
Generate synthetic multi-source financial data for testing the full pipeline.

Creates realistic dummy data with:
    - Daily market data (prices, volume, etc.)
    - Monthly macro data
    - Weekly sentiment data
    - Quarterly financial data
    - Text embeddings (128-dim)

Usage:
    python generate_sample_data.py --output-dir data/raw --years 8
"""

import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime, timedelta


def trading_days(start: str, end: str) -> pd.DatetimeIndex:
    """Generate business day date range."""
    return pd.bdate_range(start=start, end=end)


def add_noise(series: np.ndarray, noise_std: float = 0.01) -> np.ndarray:
    """Add small Gaussian noise."""
    return series + np.random.randn(len(series)) * noise_std


def generate_market_data(dates: pd.DatetimeIndex) -> pd.DataFrame:
    """Daily market data: prices, volume, valuations."""
    n = len(dates)
    t = np.arange(n)

    # Simulate a price series with trend + cycles + noise
    trend = 0.0003 * t  # slight upward drift
    cycle1 = 0.1 * np.sin(2 * np.pi * t / 252)       # annual cycle
    cycle2 = 0.05 * np.sin(2 * np.pi * t / 63)        # quarterly cycle
    noise = np.random.randn(n) * 0.01
    log_returns = trend + noise
    log_returns[1:] += cycle1[1:] - cycle1[:-1] + cycle2[1:] - cycle2[:-1]
    close = 3000 * np.exp(np.cumsum(log_returns))

    df = pd.DataFrame({
        "datetime": dates,
        "close": add_noise(close, 0.005),
        "open": add_noise(close * (1 + np.random.randn(n) * 0.003), 0.005),
        "high": add_noise(close * (1 + np.abs(np.random.randn(n)) * 0.01), 0.005),
        "low": add_noise(close * (1 - np.abs(np.random.randn(n)) * 0.01), 0.005),
        "volume": add_noise(np.abs(1e9 + 5e8 * np.sin(2 * np.pi * t / 63) + np.random.randn(n) * 2e8), 1e7),
        "pe_ratio": add_noise(15 + 3 * np.sin(2 * np.pi * t / 504), 0.1),
        "turnover": add_noise(0.02 + 0.01 * np.sin(2 * np.pi * t / 126) + np.random.randn(n) * 0.003, 0.001),
        "volatility_20d": add_noise(0.2 + 0.05 * np.sin(2 * np.pi * t / 63) + np.random.randn(n) * 0.02, 0.005),
        "margin_balance": add_noise(np.abs(1e11 + 2e10 * np.sin(2 * np.pi * t / 252)) * (1 + np.random.randn(n) * 0.01), 1e9),
        "northbound_flow": add_noise(1e8 * np.sin(2 * np.pi * t / 21) * (1 + np.random.randn(n) * 0.5), 1e7),
    })

    # Melt to long format
    id_vars = ["datetime"]
    value_vars = [c for c in df.columns if c != "datetime"]
    long = df.melt(id_vars=id_vars, value_vars=value_vars,
                   var_name="variable", value_name="value")
    long["source"] = "market"
    return long[["datetime", "source", "variable", "value"]]


def generate_macro_data(dates: pd.DatetimeIndex) -> pd.DataFrame:
    """Monthly macro-economic indicators."""
    # Resample to month-end
    monthly = dates.to_series().resample("ME").last().dropna()
    monthly_dates = pd.DatetimeIndex(monthly.values)
    n = len(monthly_dates)
    t = np.arange(n)

    data = {
        "cpi_yoy": 2.0 + 1.5 * np.sin(2 * np.pi * t / 48) + np.random.randn(n) * 0.3,
        "ppi_yoy": 1.0 + 2.0 * np.sin(2 * np.pi * t / 36) + np.random.randn(n) * 0.5,
        "pmi": 50 + 2 * np.sin(2 * np.pi * t / 24) + np.random.randn(n) * 1.0,
        "m2_yoy": 9.0 + 1.5 * np.sin(2 * np.pi * t / 40) + np.random.randn(n) * 0.3,
        "social_financing": 1.5e12 + 5e11 * np.sin(2 * np.pi * t / 12) + np.random.randn(n) * 2e11,
        "industrial_output_yoy": 5.0 + 2.0 * np.sin(2 * np.pi * t / 36) + np.random.randn(n) * 0.8,
        "fixed_asset_investment": 5.0 + 2.0 * np.sin(2 * np.pi * t / 32) + np.random.randn(n) * 0.6,
        "retail_sales_yoy": 7.0 + 2.5 * np.sin(2 * np.pi * t / 28) + np.random.randn(n) * 0.7,
        "exports_yoy": 8.0 + 5.0 * np.sin(2 * np.pi * t / 24) + np.random.randn(n) * 1.5,
        "imports_yoy": 6.0 + 4.0 * np.sin(2 * np.pi * t / 24) + np.random.randn(n) * 1.2,
        "usdcny": 6.8 + 0.3 * np.sin(2 * np.pi * t / 48) + np.random.randn(n) * 0.05,
        "unemployment": 5.0 + 0.3 * np.sin(2 * np.pi * t / 36) + np.random.randn(n) * 0.1,
    }

    records = []
    for i, d in enumerate(monthly_dates):
        for var, val in data.items():
            records.append({
                "datetime": d,
                "source": "macro",
                "variable": var,
                "value": val[i],
            })
    return pd.DataFrame(records)


def generate_sentiment_data(dates: pd.DatetimeIndex) -> pd.DataFrame:
    """Weekly sentiment indicators."""
    weekly = dates.to_series().resample("W-FRI").last().dropna()
    weekly_dates = pd.DatetimeIndex(weekly.values)
    n = len(weekly_dates)
    t = np.arange(n)

    data = {
        "news_polarity": np.clip(np.random.randn(n) * 0.3 + 0.05 * np.sin(2 * np.pi * t / 52), -1, 1),
        "news_volume": np.abs(1000 + 200 * np.sin(2 * np.pi * t / 26) + np.random.randn(n) * 100),
        "analyst_optimism": np.clip(np.random.randn(n) * 0.2 + 0.1 * np.sin(2 * np.pi * t / 26), -1, 1),
        "social_media_buzz": np.abs(500 + 150 * np.sin(2 * np.pi * t / 13) + np.random.randn(n) * 80),
        "report_upgrade_ratio": np.clip(0.5 + 0.1 * np.sin(2 * np.pi * t / 26) + np.random.randn(n) * 0.1, 0, 1),
    }

    records = []
    for i, d in enumerate(weekly_dates):
        for var, val in data.items():
            records.append({
                "datetime": d,
                "source": "sentiment",
                "variable": var,
                "value": val[i],
            })
    return pd.DataFrame(records)


def generate_financial_data(dates: pd.DatetimeIndex) -> pd.DataFrame:
    """Quarterly financial statement aggregates."""
    quarterly = dates.to_series().resample("QE").last().dropna()
    quarterly_dates = pd.DatetimeIndex(quarterly.values)
    n = len(quarterly_dates)
    t = np.arange(n)

    data = {
        "roe_median": 10.0 + 2.0 * np.sin(2 * np.pi * t / 16) + np.random.randn(n) * 0.5,
        "eps_growth_yoy": 8.0 + 4.0 * np.sin(2 * np.pi * t / 16) + np.random.randn(n) * 2.0,
        "revenue_growth_yoy": 7.0 + 3.0 * np.sin(2 * np.pi * t / 16) + np.random.randn(n) * 1.5,
        "debt_to_equity": 1.2 + 0.2 * np.sin(2 * np.pi * t / 20) + np.random.randn(n) * 0.05,
        "net_margin_median": 8.0 + 1.5 * np.sin(2 * np.pi * t / 16) + np.random.randn(n) * 0.3,
    }

    records = []
    for i, d in enumerate(quarterly_dates):
        for var, val in data.items():
            records.append({
                "datetime": d,
                "source": "financial",
                "variable": var,
                "value": val[i],
            })
    return pd.DataFrame(records)


def generate_text_embeddings(dates: pd.DatetimeIndex, n_dims: int = 128) -> pd.DataFrame:
    """Simulated text embedding data points (pretend we ran text_encoder.py)."""
    daily_dates = dates[::5]  # every ~5 trading days there's some news
    n = len(daily_dates)

    # Simulate PCA-reduced embeddings
    embeddings = np.random.randn(n, n_dims) * 0.5

    records = []
    for i, d in enumerate(daily_dates):
        for dim in range(n_dims):
            records.append({
                "datetime": d,
                "source": "text_embed",
                "variable": f"text_embed_{dim}",
                "value": embeddings[i, dim],
            })
    return pd.DataFrame(records)


def generate_llm_state(dates: pd.DatetimeIndex) -> pd.DataFrame:
    """Simulated LLM market state scores (pretend we ran llm_state_generator.py)."""
    monthly = dates.to_series().resample("ME").last().dropna()
    monthly_dates = pd.DatetimeIndex(monthly.values)
    n = len(monthly_dates)
    t = np.arange(n)

    dims = ["risk_appetite", "liquidity", "growth_expectation", "policy_stance"]
    cycles = [24, 30, 36, 40]
    records = []

    for i, d in enumerate(monthly_dates):
        for dim, cycle_len in zip(dims, cycles):
            val = np.clip(
                0.3 * np.sin(2 * np.pi * t[i] / cycle_len) + np.random.randn() * 0.2,
                -1.0, 1.0
            )
            records.append({
                "datetime": d,
                "source": "llm_state",
                "variable": dim,
                "value": val,
            })

    return pd.DataFrame(records)


def generate_text_summaries(dates: pd.DatetimeIndex) -> pd.DataFrame:
    """Simulated monthly text summaries for LLM state generation."""
    monthly = dates.to_series().resample("ME").last().dropna()
    monthly_dates = pd.DatetimeIndex(monthly.values)

    words_a = ["回暖", "谨慎", "乐观", "观望"]
    words_b = ["放大", "萎缩", "持平"]
    words_c = ["复苏", "放缓", "企稳", "探底"]
    words_d = ["维持宽松基调", "边际收紧信号", "中性偏松", "定向支持"]
    words_e = ["震荡走强", "回调盘整", "窄幅整理", "技术反弹"]
    words_f = ["维持利率不变", "释放鹰派信号", "释放鸽派信号", "按兵不动"]
    words_g = ["超", "低于", "符合"]
    words_h = ["新能源与半导体景气度较高", "地产链仍承压", "消费复苏分化", "基建投资提速"]
    words_flow = ["大幅流入", "小幅净流出", "持续净流入", "明显流出"]
    words_pboc = ["降准释放流动性", "逆回购净投放", "MLF超额续作"]

    summaries = []
    for i, d in enumerate(monthly_dates):
        rng = np.random.RandomState(i)
        idx = i % 3
        if idx == 0:
            summary = (
                f"本月A股市场震荡，沪深300指数变动{rng.uniform(-5,5):.1f}%。"
                f"央行{words_pboc[i%3]}操作，CPI同比{rng.uniform(1,3):.1f}%。"
                f"北向资金{words_flow[i%4]}。"
            )
        elif idx == 1:
            summary = (
                f"市场情绪{words_a[i%4]}，成交量较上月{words_b[i%3]}。"
                f"PMI录得{rng.uniform(49,52):.1f}，显示经济{words_c[i%4]}趋势。"
                f"政策层面{words_d[i%4]}。"
            )
        else:
            summary = (
                f"外围市场{words_e[i%4]}，美联储{words_f[i%4]}。"
                f"国内宏观数据{words_g[i%3]}预期，社融规模{rng.uniform(1.0,3.5):.1f}万亿。"
                f"行业层面{words_h[i%4]}。"
            )
        summaries.append({"date": d, "summary": summary})

    return pd.DataFrame(summaries)


def main():
    parser = argparse.ArgumentParser(description="Generate sample financial data")
    parser.add_argument("--output-dir", type=str, default="data/raw")
    parser.add_argument("--years", type=int, default=8,
                        help="Years of synthetic data (default 8)")
    parser.add_argument("--start-date", type=str, default="2016-01-01")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    start = pd.Timestamp(args.start_date)
    end = start + pd.DateOffset(years=args.years)
    dates = trading_days(str(start), str(end))
    print(f"Generating {len(dates)} trading days ({start.date()} → {end.date()})")

    # Generate each source
    sources: dict[str, pd.DataFrame] = {
        "market": generate_market_data(dates),
        "macro": generate_macro_data(dates),
        "sentiment": generate_sentiment_data(dates),
        "financial": generate_financial_data(dates),
        "text_embed": generate_text_embeddings(dates),
        "llm_state": generate_llm_state(dates),
    }

    for name, df in sources.items():
        path = output_dir / f"{name}.csv"
        df.to_csv(path, index=False)
        print(f"  {path}: {len(df):,} rows, {df['variable'].nunique()} vars, "
              f"freq={df['datetime'].diff().mode().iloc[0] if len(df) > 1 else 'N/A'}")

    # Text summaries for LLM state generation
    summaries = generate_text_summaries(dates)
    summaries_path = output_dir / "monthly_summaries.csv"
    summaries.to_csv(summaries_path, index=False)
    print(f"  {summaries_path}: {len(summaries)} monthly summaries")

    # Generate news text file for text_encoder
    news_dates = dates[::5]
    news = pd.DataFrame({
        "date": news_dates,
        "headline": [f"市场日报 {d.strftime('%Y-%m-%d')}: 沪深两市{'上涨' if i%3!=0 else '调整'}"
                     for i, d in enumerate(news_dates)],
    })
    news_path = output_dir / "news_headlines.csv"
    news.to_csv(news_path, index=False)
    print(f"  {news_path}: {len(news)} news headlines")

    print("\nDone. Run the pipeline:")
    print("  1. python data/prepare_data.py")
    print("  2. python utils/text_encoder.py --input data/raw/news_headlines.csv --text-col headline")
    print("  3. python utils/llm_state_generator.py --input data/raw/monthly_summaries.csv")
    print("  4. python utils/concept_similarity.py")
    print("  5. python train.py")


if __name__ == "__main__":
    main()
