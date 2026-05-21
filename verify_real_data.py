"""Quick verification: real data pipeline → model forward pass."""
import sys; sys.path.insert(0, '.')
import torch
from torch.utils.data import DataLoader
from utils.dataset import SlidingWindowDataset, collate_fn
from model import FinancialMoETransformer
from config import model_cfg

print("Loading dataset...")
ds = SlidingWindowDataset(
    data_path='data/processed/data_points.parquet',
    labels_path='data/processed/labels.parquet',
    window_days=365, forecast_horizon=21, max_seq_len=model_cfg.max_seq_len,
    use_cache=False
)
print(f'Windows: {len(ds)}, Vocab: {len(ds.var_to_id)}')

print("\nBuilding model...")
model = FinancialMoETransformer(
    vocab_size=max(model_cfg.vocab_size, len(ds.var_to_id) + 2),
    d_model=model_cfg.d_model, nhead=model_cfg.nhead,
    num_layers=model_cfg.num_layers, dim_feedforward=model_cfg.dim_feedforward,
    attn_type=model_cfg.attn_type, num_experts=model_cfg.num_experts,
    expert_hidden=model_cfg.expert_hidden, top_k=model_cfg.top_k,
)
n = sum(p.numel() for p in model.parameters())
print(f'Parameters: {n:,}')

print("\nTesting single sample...")
sample = ds[0]
pred, lb = model(
    sample['values'].unsqueeze(0), sample['var_ids'].unsqueeze(0),
    sample['day'].unsqueeze(0), sample['month'].unsqueeze(0),
    sample['dow'].unsqueeze(0), sample['year_offset'].unsqueeze(0),
    sample['time_since_update'].unsqueeze(0), sample['mask'].unsqueeze(0)
)
print(f'  pred={pred.item():.4f}, lb_loss={lb.item():.4f}')

print("\nTesting batch...")
loader = DataLoader(ds, batch_size=2, shuffle=True, collate_fn=collate_fn)
batch = next(iter(loader))
pred2, lb2 = model(
    batch['values'], batch['var_ids'], batch['day'], batch['month'],
    batch['dow'], batch['year_offset'], batch['time_since_update'],
    batch['mask'].bool()
)
mse = torch.nn.functional.mse_loss(pred2.squeeze(-1), batch['label'])
print(f'  MSE={mse.item():.4f}, LB={lb2.item():.4f}')

print("\n Testing training step...")
model.train()
optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
loss = mse + 0.01 * lb2
loss.backward()
optimizer.step()
optimizer.zero_grad()
print(f'  Loss={loss.item():.4f}, Grad OK')

print("\n" + "="*50)
print("  PIPELINE VERIFIED — READY TO TRAIN")
print("="*50)
