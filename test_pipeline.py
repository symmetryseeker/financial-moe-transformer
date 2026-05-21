"""End-to-end pipeline test: dataset -> model -> training step."""
import sys
sys.path.insert(0, '.')

import torch
from torch.utils.data import DataLoader

from utils.dataset import SlidingWindowDataset, collate_fn
from model import FinancialMoETransformer
from config import model_cfg, train_cfg

# ── Dataset ────────────────────────────────────────────────────────────────
print("=" * 50)
print("1. Testing Dataset")
print("=" * 50)

ds = SlidingWindowDataset(
    data_path='data/processed/data_points.parquet',
    labels_path='data/processed/labels.parquet',
    window_days=365,
    forecast_horizon=21,
    max_seq_len=2048,
    use_cache=False,
)
print(f'Vocab size: {len(ds.var_to_id)}')
print(f'Windows: {len(ds.window_dates)}')

sample = ds[0]
for k, v in sample.items():
    if hasattr(v, 'shape'):
        print(f'  {k}: shape={list(v.shape)}, dtype={v.dtype}')
    else:
        print(f'  {k}: {v}')

# ── DataLoader ─────────────────────────────────────────────────────────────
print(f"\n{'='*50}")
print("2. Testing DataLoader")
print("=" * 50)

train_dates, val_dates, test_dates = ds.train_val_test_split()
train_idx = [i for i, d in enumerate(ds.window_dates) if d in train_dates]
print(f'Train windows: {len(train_idx)}, Val: {len(val_dates)}, Test: {len(test_dates)}')

loader = DataLoader(ds, batch_size=4, shuffle=True, collate_fn=collate_fn)
batch = next(iter(loader))
for k, v in batch.items():
    print(f'  {k}: {list(v.shape)}')

# ── Model ──────────────────────────────────────────────────────────────────
print(f"\n{'='*50}")
print("3. Testing Forward + Backward Pass")
print("=" * 50)

model = FinancialMoETransformer(
    vocab_size=max(model_cfg.vocab_size, len(ds.var_to_id) + 2),
    d_model=model_cfg.d_model,
    nhead=model_cfg.nhead,
    num_layers=model_cfg.num_layers,
    dim_feedforward=model_cfg.dim_feedforward,
    dropout=model_cfg.dropout,
    attn_dropout=model_cfg.attn_dropout,
    attn_type=model_cfg.attn_type,
    num_experts=model_cfg.num_experts,
    expert_hidden=model_cfg.expert_hidden,
    top_k=model_cfg.top_k,
)
n_params = sum(p.numel() for p in model.parameters())
print(f'Parameters: {n_params:,}')

# Forward pass
model.train()
pred, lb_loss = model(
    batch['values'],
    batch['var_ids'],
    batch['day'],
    batch['month'],
    batch['dow'],
    batch['year_offset'],
    batch['time_since_update'],
    batch['mask'].bool(),
)
print(f'Pred: {pred.shape}, LB loss: {lb_loss.item():.4f}')

# Compute loss and backward
mse = torch.nn.functional.mse_loss(pred.squeeze(-1), batch['label'])
loss = train_cfg.mse_coef * mse + train_cfg.load_balance_coef * lb_loss
loss.backward()
print(f'MSE: {mse.item():.4f}, Total: {loss.item():.4f}')

# Check gradients flow
grad_norms = {}
for name, param in model.named_parameters():
    if param.grad is not None:
        g = param.grad.norm().item()
        if g > 0:
            grad_norms[name] = g
print(f'Parameters with gradients: {len(grad_norms)}')

# ── Training step (gradient accumulation simulator) ────────────────────────
print(f"\n{'='*50}")
print("4. Simulating Mini Training Loop")
print("=" * 50)

from torch.optim import AdamW
optimizer = AdamW(model.parameters(), lr=3e-4)

losses = []
for step in range(10):
    # Get a new batch
    batch = next(iter(loader))
    pred, lb_loss = model(
        batch['values'],
        batch['var_ids'],
        batch['day'],
        batch['month'],
        batch['dow'],
        batch['year_offset'],
        batch['time_since_update'],
        batch['mask'].bool(),
    )
    mse = torch.nn.functional.mse_loss(pred.squeeze(-1), batch['label'])
    loss = train_cfg.mse_coef * mse + train_cfg.load_balance_coef * lb_loss

    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()

    losses.append(loss.item())
    if step % 3 == 0:
        print(f'  Step {step}: loss={loss.item():.4f}, mse={mse.item():.4f}')

print(f'Loss trend: {losses[0]:.4f} -> {losses[-1]:.4f}')

print(f"\n{'='*50}")
print("ALL TESTS PASSED")
print("=" * 50)
