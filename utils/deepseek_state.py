"""
Generate monthly market state features using DeepSeek V4 API.

Cost analysis:
  - Each prompt: ~400 tokens (market summary paragraph)
  - Each response: ~60 tokens (JSON with 4 numbers)
  - For 120 months: ~55K tokens total
  - DeepSeek V4 pricing: ~$0.14/M input, ~$0.28/M output
  - Total cost: < $0.01 (negligible)

Requires:
  pip install openai  (DeepSeek API is OpenAI-compatible)

Usage:
  python utils/deepseek_state.py --api-key YOUR_DEEPSEEK_API_KEY
"""

import json
import time
import logging
import argparse
from pathlib import Path
import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)

# DeepSeek API endpoint (OpenAI-compatible)
DEEPSEEK_BASE_URL = "https://api.deepseek.com/v1"
DEEPSEEK_MODEL = "deepseek-chat"  # DeepSeek V4

SYSTEM_PROMPT = """你是一位资深中国宏观分析师。根据提供的当月中国市场摘要，输出一个JSON对象，包含四个维度的评分，每个维度从 -1.0（极度悲观/紧缩）到 +1.0（极度乐观/宽松），使用一位小数精度：

- risk_appetite: 市场风险偏好（-1恐慌避险，+1积极追涨）
- liquidity: 流动性状况（-1极度紧缩，+1极度宽松）
- growth_expectation: 经济增长预期（-1严重衰退，+1强劲复苏）
- policy_stance: 政策立场（-1强力紧缩，+1强力刺激）

只输出JSON，不要附加解释。格式示例：
{"risk_appetite": 0.3, "liquidity": 0.5, "growth_expectation": -0.2, "policy_stance": 0.1}"""


def build_monthly_summary(month: pd.Timestamp, data_dir: Path) -> str:
    """
    Build a natural language summary for a given month from available data.

    Extracts key metrics (index returns, macro data, etc.) and writes
    a paragraph describing the month's market conditions.
    """
    parts = [f"{month.year}年{month.month}月中国市场摘要：\n"]

    # 1. Index performance
    csi300 = _read_latest(data_dir, "market", "csi300_daily.csv", "date", month)
    if csi300 is not None and len(csi300) > 0:
        month_data = csi300[
            (pd.to_datetime(csi300["date"]).dt.year == month.year) &
            (pd.to_datetime(csi300["date"]).dt.month == month.month)
        ]
        if len(month_data) >= 10:
            ret = (month_data["close"].iloc[-1] / month_data["close"].iloc[0] - 1) * 100
            direction = "上涨" if ret > 0 else "下跌"
            parts.append(f"沪深300指数当月{direction}{abs(ret):.1f}%。")

    # 2. Macro indicators
    for fname, label, col in [
        ("pmi.csv", "制造业PMI", "制造业-指数"),
        ("money_supply.csv", "M2同比", "m2_yoy"),
        ("cpi_yearly.csv", "CPI同比", "现值"),
        ("bond_yield_curve.csv", "10年期国债收益率", "10年"),
    ]:
        val = _read_macro_value(data_dir, fname, label, col, month)
        if val is not None:
            parts.append(f"{label}约{val:.1f}。")

    parts.append("\n请根据上述信息输出该月市场状态四维评分JSON。")
    return "".join(parts)


def _read_latest(data_dir, category, filename, date_col, month):
    path = data_dir / category / filename
    if not path.exists():
        return None
    try:
        return pd.read_csv(path)
    except Exception:
        return None


def _read_macro_value(data_dir, filename, label, col, month):
    """Read the most recent macro value for a given month."""
    for cat in ["macro", ""]:
        path = data_dir / cat / filename if cat else data_dir / filename
        if not path.exists():
            continue
        try:
            df = pd.read_csv(path)
            if "月份" in df.columns:
                df["date"] = pd.to_datetime(df["月份"].astype(str) + "01", format="%Y%m%d", errors="coerce")
            elif "month" in df.columns:
                df["date"] = pd.to_datetime(df["month"].astype(str), format="%Y%m", errors="coerce")
            elif "date" in df.columns:
                df["date"] = pd.to_datetime(df["date"], errors="coerce")
            else:
                continue

            if "date" not in df.columns:
                continue

            df = df.dropna(subset=["date"])
            df = df[df["date"] <= month]
            if df.empty:
                continue

            # Find the column with the data
            for c in df.columns:
                if col in str(c):
                    val = df.iloc[-1][c]
                    if pd.notna(val):
                        return float(val)
        except Exception:
            pass
    return None


def generate_states(data_dir: str = "D:/financial_data",
                    api_key: str = None,
                    model: str = DEEPSEEK_MODEL,
                    base_url: str = DEEPSEEK_BASE_URL,
                    start_year: int = 2015,
                    end_year: int = 2026):
    """
    Generate market state scores for each month using DeepSeek V4.

    Args:
        data_dir: path to financial_data directory
        api_key: DeepSeek API key
        model: model name
        base_url: API base URL
        start_year, end_year: year range
    """
    from openai import OpenAI

    client = OpenAI(api_key=api_key, base_url=base_url)
    data_dir = Path(data_dir)

    months = []
    for year in range(start_year, end_year + 1):
        for month in range(1, 13):
            m = pd.Timestamp(year=year, month=month, day=1)
            if m > pd.Timestamp.now():
                break
            months.append(m)

    logger.info(f"Generating market states for {len(months)} months")
    logger.info(f"Model: {model}, Estimated cost: < $0.01")

    records = []
    total_tokens = 0

    for i, month in enumerate(months):
        summary = build_monthly_summary(month, data_dir)

        try:
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": summary},
                ],
                temperature=0.1,
                max_tokens=100,
            )

            content = response.choices[0].message.content.strip()
            total_tokens += response.usage.total_tokens

            # Parse JSON
            state = _parse_json(content)
            for dim, val in state.items():
                records.append({
                    "datetime": month,
                    "source": "llm_state",
                    "variable": dim,
                    "value": float(np.clip(val, -1.0, 1.0)),
                    "time_since_update": 0.0,
                })

        except Exception as e:
            logger.warning(f"  {month.strftime('%Y-%m')}: FAILED ({e})")
            # Fallback: neutral state
            for dim in ["risk_appetite", "liquidity", "growth_expectation", "policy_stance"]:
                records.append({
                    "datetime": month,
                    "source": "llm_state",
                    "variable": dim,
                    "value": 0.0,
                    "time_since_update": 0.0,
                })

        if (i + 1) % 12 == 0:
            logger.info(f"  {i+1}/{len(months)} months done ({total_tokens} tokens used)")

        time.sleep(0.3)  # rate limit

    # Save
    outdir = Path(data_dir) / "text"
    outdir.mkdir(parents=True, exist_ok=True)
    out = pd.DataFrame(records)
    outpath = outdir / "llm_state_deepseek.csv"
    out.to_csv(outpath, index=False)

    logger.info(f"Saved {len(out)} rows to {outpath}")
    logger.info(f"Total tokens: {total_tokens:,} (cost: ~${total_tokens/1e6*0.2:.4f})")

    # Stats
    for dim in ["risk_appetite", "liquidity", "growth_expectation", "policy_stance"]:
        vals = out[out.variable == dim]["value"]
        logger.info(f"  {dim}: mean={vals.mean():.3f}, std={vals.std():.3f}")

    return out


def _parse_json(text: str) -> dict:
    """Extract JSON from model output."""
    import re
    # Try direct parse
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Try code fence
    match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(1))
        except json.JSONDecodeError:
            pass

    # Try to find any JSON object
    match = re.search(r'\{.*?\}', text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            pass

    logger.warning(f"Could not parse JSON from: {text[:100]}")
    return {"risk_appetite": 0.0, "liquidity": 0.0,
            "growth_expectation": 0.0, "policy_stance": 0.0}


def main():
    parser = argparse.ArgumentParser(description="DeepSeek Market State Generator")
    parser.add_argument("--api-key", type=str, required=True,
                        help="DeepSeek API key (https://platform.deepseek.com)")
    parser.add_argument("--data-dir", type=str, default="D:/financial_data")
    parser.add_argument("--start-year", type=int, default=2015)
    parser.add_argument("--end-year", type=int, default=2026)
    parser.add_argument("--model", type=str, default=DEEPSEEK_MODEL)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s",
                        datefmt="%H:%M:%S")

    generate_states(
        data_dir=args.data_dir,
        api_key=args.api_key,
        model=args.model,
        start_year=args.start_year,
        end_year=args.end_year,
    )


if __name__ == "__main__":
    main()
