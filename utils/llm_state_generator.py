"""
Generate monthly market-state features using a quantised LLM.

Uses 4-bit Qwen2.5-0.5B-Instruct to produce four-dimensional state scores
from concatenated monthly summaries (news headlines, macro events).

Output (per month):
    risk_appetite, liquidity, growth_expectation, policy_stance
    each in [-1, 1] range.

Usage:
    python utils/llm_state_generator.py \
        --input data/raw/monthly_summaries.csv \
        --output data/embeddings/llm_state.parquet
"""

import argparse
import json
import re
import numpy as np
import pandas as pd
from pathlib import Path
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig


SYSTEM_PROMPT = """你是一位资深宏观分析师。根据当月的中国市场摘要，输出一个JSON对象，包含四个维度的评分，每个维度从-1（极度悲观/紧缩）到+1（极度乐观/宽松）：

- risk_appetite: 市场风险偏好
- liquidity: 流动性状况
- growth_expectation: 经济增长预期
- policy_stance: 政策立场

只输出JSON，不要附加任何解释。"""


def build_prompt(monthly_summary: str) -> str:
    return f"{SYSTEM_PROMPT}\n\n当月市场摘要：\n{monthly_summary}\n\n请输出JSON："


def extract_json(text: str) -> dict | None:
    """Extract JSON object from model output, handling markdown fences."""
    # Try to find JSON in code fences
    match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', text, re.DOTALL)
    if match:
        text = match.group(1)
    # Try direct JSON
    match = re.search(r'\{.*?\}', text, re.DOTALL)
    if not match:
        return None
    try:
        return json.loads(match.group())
    except json.JSONDecodeError:
        return None


def load_quantized_model(model_name: str = "Qwen/Qwen2.5-0.5B-Instruct"):
    """Load model in 4-bit quantisation for low VRAM usage."""
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
    )

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
        torch_dtype=torch.float16,
    )
    return model, tokenizer


def generate_state(model, tokenizer, prompt: str) -> dict:
    """Generate four-dim state from a single monthly summary."""
    messages = [{"role": "user", "content": prompt}]
    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )

    inputs = tokenizer(text, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=128,
            temperature=0.1,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
        )

    response = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
    result = extract_json(response)

    if result is None:
        print(f"Failed to parse JSON from: {response[:200]}")
        return {"risk_appetite": 0.0, "liquidity": 0.0,
                "growth_expectation": 0.0, "policy_stance": 0.0}

    # Validate and clamp
    expected_keys = ["risk_appetite", "liquidity", "growth_expectation", "policy_stance"]
    for k in expected_keys:
        if k not in result:
            result[k] = 0.0
        result[k] = float(np.clip(result[k], -1.0, 1.0))
    return result


def main():
    parser = argparse.ArgumentParser(description="LLM market state generation")
    parser.add_argument("--input", type=str, required=True,
                        help="CSV with columns: date, summary")
    parser.add_argument("--date-col", type=str, default="date")
    parser.add_argument("--summary-col", type=str, default="summary")
    parser.add_argument("--output", type=str, required=True,
                        help="Output parquet (long format for data-points table)")
    parser.add_argument("--model-name", type=str,
                        default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--device", type=str, default="cuda",
                        help="Device for model (cuda/cpu)")
    args = parser.parse_args()

    # Load data
    input_path = Path(args.input)
    df = pd.read_csv(input_path) if input_path.suffix == ".csv" else pd.read_parquet(input_path)
    print(f"Loaded {len(df)} monthly summaries")

    # Load model (4-bit if CUDA available, otherwise CPU float32)
    if args.device == "cuda" and torch.cuda.is_available():
        model, tokenizer = load_quantized_model(args.model_name)
        print(f"Loaded 4-bit {args.model_name} on GPU")
    else:
        print("Loading model on CPU (no quantisation)...")
        tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            args.model_name,
            torch_dtype=torch.float32,
            trust_remote_code=True,
        )
        model = model.to(args.device)
        tokenizer.pad_token = tokenizer.eos_token

    # Generate state for each month
    records = []
    for _, row in df.iterrows():
        summary = row[args.summary_col]
        if pd.isna(summary) or not str(summary).strip():
            state = {"risk_appetite": 0.0, "liquidity": 0.0,
                     "growth_expectation": 0.0, "policy_stance": 0.0}
        else:
            prompt = build_prompt(str(summary))
            state = generate_state(model, tokenizer, prompt)

        date_val = row[args.date_col]
        for dim_name, dim_val in state.items():
            records.append({
                "datetime": pd.to_datetime(date_val),
                "source": "llm_state",
                "variable": dim_name,
                "value": dim_val,
                "time_since_update": 0.0,
            })

    out = pd.DataFrame(records)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(output_path, index=False)
    print(f"Saved {len(out)} rows to {args.output}")

    # Quick stats
    for dim in ["risk_appetite", "liquidity", "growth_expectation", "policy_stance"]:
        vals = out[out.variable == dim]["value"]
        print(f"  {dim}: mean={vals.mean():.3f}, std={vals.std():.3f}")


if __name__ == "__main__":
    main()
