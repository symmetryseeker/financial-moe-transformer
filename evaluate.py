"""
Rolling backtest evaluation.

Generates monthly predictions using a sliding 1-year window, then computes:
    - IC (rank correlation between prediction and realised return)
    - R²
    - Long/short portfolio Sharpe ratio
    - Expert gate weight time series

Usage:
    python evaluate.py --checkpoint checkpoints/best.pt --output results/
"""

import argparse
import sys
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).parent))

from config import model_cfg, train_cfg, ROOT
from model import FinancialMoETransformer
from utils.dataset import SlidingWindowDataset, collate_fn


@torch.no_grad()
def rolling_predict(model, dataset, device, batch_size: int = 8):
    """
    Generate predictions for all windows in the dataset.

    Returns:
        df: DataFrame with columns [datetime, pred, label, *expert_weights]
    """
    model.eval()
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=0,
        pin_memory=True,
    )

    records = []
    for i, batch in enumerate(loader):
        if not batch:
            continue

        batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                 for k, v in batch.items()}

        pred, _ = model(
            batch["values"], batch["var_ids"],
            batch["day"], batch["month"], batch["dow"],
            batch["year_offset"], batch["time_since_update"],
            batch["mask"].bool() if "mask" in batch else None,
        )

        expert_weights = model.get_expert_weights(
            batch["values"], batch["var_ids"],
            batch["day"], batch["month"], batch["dow"],
            batch["year_offset"], batch["time_since_update"],
            batch["mask"].bool() if "mask" in batch else None,
        )

        preds = pred.cpu().numpy().flatten()
        labels = batch["label"].cpu().numpy().flatten()
        gates = expert_weights.cpu().numpy()

        # Get window end dates for this batch
        batch_start = i * batch_size
        for j in range(len(preds)):
            idx = batch_start + j
            if idx < len(dataset.window_dates):
                rec = {
                    "datetime": dataset.window_dates[idx],
                    "pred": preds[j],
                    "label": labels[j],
                }
                for e in range(gates.shape[1]):
                    rec[f"expert_{e}_weight"] = gates[j, e]
                records.append(rec)

    return pd.DataFrame(records)


def compute_metrics(df: pd.DataFrame) -> dict:
    """Compute evaluation metrics from predictions."""
    df = df.dropna(subset=["pred", "label"])

    # Rank IC
    ic = df["pred"].corr(df["label"])
    rank_ic = df["pred"].rank().corr(df["label"].rank())

    # R²
    ss_res = ((df["label"] - df["pred"]) ** 2).sum()
    ss_tot = ((df["label"] - df["label"].mean()) ** 2).sum()
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0

    # Long/short strategy
    # Each month: go long top quintile, short bottom quintile
    df = df.sort_values("datetime").copy()
    df["month"] = df["datetime"].dt.to_period("M")
    returns = []
    for _, grp in df.groupby("month"):
        if len(grp) < 5:
            continue
        q80 = grp["pred"].quantile(0.8)
        q20 = grp["pred"].quantile(0.2)
        long_r = grp[grp["pred"] >= q80]["label"].mean()
        short_r = grp[grp["pred"] <= q20]["label"].mean()
        returns.append(long_r - short_r)

    returns = np.array(returns)
    sharpe = returns.mean() / returns.std() * np.sqrt(12) if len(returns) > 1 and returns.std() > 0 else 0.0
    hit_rate = (returns > 0).mean() if len(returns) > 0 else 0.0

    # Expert weight time series
    expert_cols = [c for c in df.columns if c.startswith("expert_")]
    expert_weights = df.groupby("month")[expert_cols].mean() if expert_cols else None

    return {
        "ic": ic,
        "rank_ic": rank_ic,
        "r2": r2,
        "sharpe": sharpe,
        "hit_rate": hit_rate,
        "n_months": len(returns),
        "expert_weights": expert_weights,
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate Financial MoE Transformer")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--data-dir", type=str, default="data/processed")
    parser.add_argument("--output", type=str, default="results")
    parser.add_argument("--device", type=str, default=None)
    args = parser.parse_args()

    device = torch.device(args.device if args.device else
                          ("cuda" if torch.cuda.is_available() else "cpu"))
    print(f"Using device: {device}")

    # Load model
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    m_cfg = ckpt.get("config", model_cfg)

    model = FinancialMoETransformer(
        vocab_size=m_cfg.vocab_size, d_model=m_cfg.d_model, nhead=m_cfg.nhead,
        num_layers=m_cfg.num_layers, dim_feedforward=m_cfg.dim_feedforward,
        dropout=m_cfg.dropout, attn_dropout=m_cfg.attn_dropout,
        activation=m_cfg.activation, attn_type=m_cfg.attn_type,
        attn_chunk_size=m_cfg.attn_chunk_size,
        num_experts=m_cfg.num_experts, expert_hidden=m_cfg.expert_hidden,
        expert_layers=m_cfg.expert_layers, top_k=m_cfg.top_k,
        max_seq_len=m_cfg.max_seq_len,
    )
    model.load_state_dict(ckpt["model"])
    model = model.to(device)
    print(f"Loaded model (epoch {ckpt.get('epoch', '?')})")

    # Dataset (test portion only)
    data_dir = Path(args.data_dir)
    full_dataset = SlidingWindowDataset(
        data_path=str(data_dir / "data_points.parquet"),
        labels_path=str(data_dir / "labels.parquet"),
        window_days=365,
        forecast_horizon=21,
        max_seq_len=m_cfg.max_seq_len,
    )

    # Use the last TEST_FRAC as test set
    _, _, test_dates = full_dataset.train_val_test_split()
    test_idx = [i for i, d in enumerate(full_dataset.window_dates) if d in test_dates]

    from torch.utils.data import Subset
    test_dataset = Subset(full_dataset, test_idx)

    print(f"Test windows: {len(test_dataset)}")

    # Rolling predictions
    print("Generating predictions ...")
    results_df = rolling_predict(model, test_dataset, device)

    # Metrics
    metrics = compute_metrics(results_df)

    print(f"\n─── Evaluation Results ───")
    print(f"  IC:        {metrics['ic']:.4f}")
    print(f"  Rank IC:   {metrics['rank_ic']:.4f}")
    print(f"  R²:        {metrics['r2']:.4f}")
    print(f"  Sharpe:    {metrics['sharpe']:.4f}")
    print(f"  Hit Rate:  {metrics['hit_rate']:.2%}")
    print(f"  N Months:  {metrics['n_months']}")

    # Save results
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    results_df.to_csv(output_dir / "predictions.csv", index=False)

    # Save metrics as JSON
    metrics_serial = {k: v for k, v in metrics.items() if k != "expert_weights"}
    with open(output_dir / "metrics.json", "w") as f:
        json.dump(metrics_serial, f, indent=2, default=str)

    # Save expert weight time series
    if metrics["expert_weights"] is not None:
        metrics["expert_weights"].to_csv(output_dir / "expert_weights.csv")

    print(f"\nResults saved to {output_dir}/")
    print(f"  - predictions.csv    ({len(results_df)} rows)")
    print(f"  - metrics.json")
    print(f"  - expert_weights.csv")


if __name__ == "__main__":
    main()
