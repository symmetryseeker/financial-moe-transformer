"""Quick training speed benchmark."""
import sys, time; sys.path.insert(0, '.')
import torch
from torch.utils.data import DataLoader
from utils.dataset import SlidingWindowDataset, collate_fn
from model import FinancialMoETransformer
from config import model_cfg, train_cfg

device = torch.device("cpu")
print(f"Device: {device}")

ds = SlidingWindowDataset(
    data_path='data/processed/data_points.parquet',
    labels_path='data/processed/labels.parquet',
    window_days=365, forecast_horizon=21, max_seq_len=model_cfg.max_seq_len,
    use_cache=False
)
print(f"Windows: {len(ds)}")

model = FinancialMoETransformer(
    vocab_size=max(model_cfg.vocab_size, len(ds.var_to_id) + 2),
    d_model=model_cfg.d_model, nhead=model_cfg.nhead,
    num_layers=model_cfg.num_layers, dim_feedforward=model_cfg.dim_feedforward,
    attn_type=model_cfg.attn_type, num_experts=model_cfg.num_experts,
    expert_hidden=model_cfg.expert_hidden, top_k=model_cfg.top_k,
).to(device)

loader = DataLoader(ds, batch_size=2, shuffle=True, collate_fn=collate_fn)
optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)

# Warmup
print("\nWarmup...")
for i, batch in enumerate(loader):
    if i >= 3: break
    pred, lb = model(batch['values'], batch['var_ids'], batch['day'],
                     batch['month'], batch['dow'], batch['year_offset'],
                     batch['time_since_update'], batch['mask'].bool())
    loss = torch.nn.functional.mse_loss(pred.squeeze(-1), batch['label']) + 0.01 * lb
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()

# Benchmark
print("Benchmarking 20 batches...")
model.train()
t0 = time.time()
n_samples = 0
for i, batch in enumerate(loader):
    if i >= 20: break
    pred, lb = model(batch['values'], batch['var_ids'], batch['day'],
                     batch['month'], batch['dow'], batch['year_offset'],
                     batch['time_since_update'], batch['mask'].bool())
    loss = torch.nn.functional.mse_loss(pred.squeeze(-1), batch['label']) + 0.01 * lb
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
    n_samples += len(batch['label'])

elapsed = time.time() - t0
per_batch = elapsed / 20
per_sample = elapsed / n_samples

batches_per_epoch = len(ds) / 2  # batch_size=2
time_per_epoch = batches_per_epoch * per_batch
time_20_epochs = time_per_epoch * 20

print(f"\n{'='*50}")
print(f"  SPEED BENCHMARK (CPU: {device})")
print(f"{'='*50}")
print(f"  Per batch (2 samples):  {per_batch:.2f}s")
print(f"  Per sample:             {per_sample:.3f}s")
print(f"  Batches per epoch:      {batches_per_epoch:.0f}")
print(f"  Time per epoch:         {time_per_epoch/60:.1f} min")
print(f"  Time for 20 epochs:     {time_20_epochs/60:.1f} min ({time_20_epochs/3600:.1f} hr)")
print(f"  Early stop (avg ~8 ep): {time_20_epochs*0.4/60:.1f} min")

# GPU estimate (2-5x faster with GTX 1060 for small model)
gpu_speedup = 1.5  # conservative for GTX 1060 with small model
print(f"\n  GPU estimate (GTX 1060, {gpu_speedup}x):")
print(f"  Time for 20 epochs:     {time_20_epochs/gpu_speedup/60:.1f} min")
print(f"  Early stop (~8 ep):     {time_20_epochs*0.4/gpu_speedup/60:.1f} min")
print(f"{'='*50}")
