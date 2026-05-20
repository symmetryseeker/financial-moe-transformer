"""
Training script v3 — Dual Embedding + Hierarchical Encoder + Advanced Losses.
"""
import argparse, sys, time, os
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR

sys.path.insert(0, str(Path(__file__).parent))

from config import ROOT, MODEL_DIR, LOG_DIR
from model import FinancialMoETransformer
from utils.dataset import SlidingWindowDataset, collate_fn
from utils.losses import CombinedLoss


class EMA:
    """Exponential Moving Average of model weights for stable inference."""
    def __init__(self, model, decay=0.999):
        self.decay = decay
        self.shadow = {}
        self._backup = {}
        self._model = model
        for n, p in model.named_parameters():
            if p.requires_grad:
                self.shadow[n] = p.data.clone()

    def update(self):
        with torch.no_grad():
            for n, p in self._model.named_parameters():
                if p.requires_grad:
                    self.shadow[n].mul_(self.decay).add_(p.data, alpha=1 - self.decay)

    def apply_shadow(self):
        for n, p in self._model.named_parameters():
            if p.requires_grad:
                self._backup[n] = p.data.clone()
                p.data = self.shadow[n]

    def restore(self):
        for n, p in self._model.named_parameters():
            if p.requires_grad and n in self._backup:
                p.data = self._backup[n]
        self._backup.clear()


def token_mask_augmentation(batch, mask_prob=0.05):
    if mask_prob <= 0: return batch
    m = batch["mask"].bool()
    rand = torch.rand_like(batch["values"].squeeze(-1))
    tm = (rand < mask_prob) & m
    batch = {k: v.clone() if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
    batch["values"] = batch["values"].clone()
    batch["values"][tm] = 0.0
    return batch


def train_epoch(model, loader, optimizer, scheduler, device, cfg, loss_fn):
    model.train()
    history = []
    optimizer.zero_grad()
    for step, batch in enumerate(loader):
        if not batch: continue
        batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
        batch = token_mask_augmentation(batch, cfg.token_mask_prob)

        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=cfg.use_amp):
            pred, lb_loss = model(
                batch["values"], batch["company_ids"], batch["metric_ids"],
                batch["day"], batch["month"], batch["dow"],
                batch["year_offset"], batch["time_since_update"],
                batch["mask"].bool(),
                batch["source_ids"], batch["time_bins"],
            )
            loss, loss_dict = loss_fn(pred, batch["label"], lb_loss)
            loss = loss / cfg.gradient_accumulation

        if cfg.use_amp:
            raise NotImplementedError("AMP not configured for this setup")
        else:
            loss.backward()

        if (step + 1) % cfg.gradient_accumulation == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.max_grad_norm)
            optimizer.step()
            optimizer.zero_grad()
            scheduler.step()

        history.append(loss_dict)

        if (step + 1) % 100 == 0:
            print(f"  batch {step+1}/{len(loader)} | MSE={loss_dict.get('mse',0):.3f}", flush=True)

    return {k: np.mean([h[k] for h in history]) for k in history[0]}


@torch.no_grad()
def validate(model, loader, device):
    model.eval()
    all_preds, all_labels = [], []
    for batch in loader:
        if not batch: continue
        batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
        pred, _ = model(
            batch["values"], batch["company_ids"], batch["metric_ids"],
            batch["day"], batch["month"], batch["dow"],
            batch["year_offset"], batch["time_since_update"],
            batch["mask"].bool(),
            batch["source_ids"], batch["time_bins"],
        )
        all_preds.append(pred.cpu().numpy())
        all_labels.append(batch["label"].cpu().numpy())

    preds = np.concatenate(all_preds).flatten()
    labels = np.concatenate(all_labels).flatten()
    ic = np.corrcoef(preds, labels)[0, 1] if len(preds) > 1 else 0.0
    rank_ic = np.corrcoef(preds.argsort().argsort(), labels.argsort().argsort())[0,1] if len(preds) > 1 else 0.0
    mse = float(np.mean((preds - labels) ** 2))
    return {"ic": float(ic), "rank_ic": float(rank_ic), "mse": mse}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--resume", type=str, default=None)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    dataset = SlidingWindowDataset(max_seq_len=8192)
    train_dates, val_dates, test_dates = dataset.train_val_test_split()
    train_idx = [i for i, d in enumerate(dataset.window_dates) if d in train_dates]
    val_idx = [i for i, d in enumerate(dataset.window_dates) if d in val_dates]
    print(f"Windows: {len(dataset)} | Train: {len(train_idx)} | Val: {len(val_idx)}")

    train_loader = DataLoader(Subset(dataset, train_idx), batch_size=args.batch_size,
        shuffle=True, collate_fn=collate_fn, num_workers=0, pin_memory=False, drop_last=True)
    val_loader = DataLoader(Subset(dataset, val_idx), batch_size=args.batch_size,
        shuffle=False, collate_fn=collate_fn, num_workers=0, pin_memory=False)

    from config import model_cfg, train_cfg

    model = FinancialMoETransformer(
        n_companies=dataset.n_companies + 2, n_metrics=dataset.n_metrics + 2,
        d_model=model_cfg.d_model, nhead=model_cfg.nhead,
        num_layers=model_cfg.num_layers, dim_feedforward=model_cfg.dim_feedforward,
        dropout=model_cfg.dropout, attn_type=model_cfg.attn_type,
        attn_chunk_size=model_cfg.attn_chunk_size,
        num_experts=model_cfg.num_experts, expert_hidden=model_cfg.expert_hidden,
        top_k=model_cfg.top_k,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model: {n_params:,} params ({dataset.n_companies:,}c x {dataset.n_metrics:,}m)")

    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=0.1)
    steps_per_epoch = len(train_loader)
    total_steps = steps_per_epoch * args.epochs
    warmup_steps = min(500, total_steps // 5)
    warmup = LinearLR(optimizer, start_factor=1e-3, total_iters=warmup_steps)
    cosine = CosineAnnealingLR(optimizer, T_max=total_steps - warmup_steps, eta_min=args.lr * 0.01)
    scheduler = SequentialLR(optimizer, schedulers=[warmup, cosine], milestones=[warmup_steps])

    loss_fn = CombinedLoss(mse_coef=1.0, load_balance_coef=train_cfg.load_balance_coef,
                           rank_coef=train_cfg.rank_coef, direction_coef=train_cfg.direction_coef)
    ema = EMA(model, decay=0.999)
    best_ic = -float("inf")
    patience = 0

    for epoch in range(args.epochs):
        t0 = time.time()
        train_losses = train_epoch(model, train_loader, optimizer, scheduler,
                                   device, train_cfg, loss_fn)
        sys.stdout.flush()
        ema.update()

        # Validate with EMA weights
        ema.apply_shadow()
        val_metrics = validate(model, val_loader, device)
        ema.restore()

        elapsed = time.time() - t0
        print(f"Ep {epoch+1:2d} ({elapsed:.0f}s) | "
              f"MSE={train_losses.get('mse',0):.3f} "
              f"Rank={train_losses.get('rank',0):.3f} "
              f"Dir={train_losses.get('direction',0):.3f} | "
              f"Val IC={val_metrics['ic']:.4f}", flush=True)

        if val_metrics["ic"] > best_ic:
            best_ic = val_metrics["ic"]
            patience = 0
            # Save EMA weights (not raw weights)
            ema.apply_shadow()
            torch.save({"epoch": epoch, "model": model.state_dict(), "best_ic": best_ic},
                       MODEL_DIR / "best.pt")
            ema.restore()
            print(f"  -> Best (IC={best_ic:.4f})")
        else:
            patience += 1

        if patience >= 5:
            print(f"Early stop at epoch {epoch+1}")
            break

    print(f"Done. Best IC: {best_ic:.4f} | Model: {MODEL_DIR / 'best.pt'}")


if __name__ == "__main__":
    main()
