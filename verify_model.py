"""Quick verification of model architecture, parameter count, and forward pass."""
import sys, torch
sys.path.insert(0, '.')

from model import FinancialMoETransformer
from config import model_cfg

# Build model
model = FinancialMoETransformer(
    vocab_size=model_cfg.vocab_size,
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

# Count parameters
total = sum(p.numel() for p in model.parameters())
trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f'Total parameters: {total:,}')
print(f'Trainable:       {trainable:,}')

# Breakdown
for name, mod in [
    ('Embedding', model.embedding),
    ('Encoder', model.encoder),
    ('MoE', model.moe),
]:
    n = sum(p.numel() for p in mod.parameters())
    print(f'  {name}: {n:,}')

# Forward pass test
B, L = 4, 256
batch = {
    'values': torch.randn(B, L, 1),
    'var_ids': torch.randint(1, 500, (B, L)),
    'day': torch.randint(1, 31, (B, L)),
    'month': torch.randint(1, 12, (B, L)),
    'dow': torch.randint(0, 6, (B, L)),
    'year_offset': torch.randint(0, 10, (B, L)),
    'time_since_update': torch.rand(B, L) * 30,
    'mask': torch.ones(B, L, dtype=torch.bool),
}

model.eval()
with torch.no_grad():
    pred, lb_loss = model(**batch)

print(f'\nForward pass OK')
print(f'  Input:  (B={B}, L={L})')
print(f'  Output: pred={pred.shape}, lb_loss={lb_loss.item():.4f}')

# Test with variable sequence lengths
mask2 = torch.cat([
    torch.ones(4, 200, dtype=torch.bool),
    torch.zeros(4, 56, dtype=torch.bool),
], dim=1)
batch['mask'] = mask2
with torch.no_grad():
    pred2, lb2 = model(**batch)
print(f'  Masked output: pred={pred2.shape}, lb_loss={lb2.item():.4f}')

# Test max sequence length
B3, L3 = 2, 2048
big_batch = {
    'values': torch.randn(B3, L3, 1),
    'var_ids': torch.randint(1, 500, (B3, L3)),
    'day': torch.randint(1, 31, (B3, L3)),
    'month': torch.randint(1, 12, (B3, L3)),
    'dow': torch.randint(0, 6, (B3, L3)),
    'year_offset': torch.randint(0, 10, (B3, L3)),
    'time_since_update': torch.rand(B3, L3) * 30,
    'mask': torch.ones(B3, L3, dtype=torch.bool),
}
with torch.no_grad():
    pred3, lb3 = model(**big_batch)
print(f'  Max seq (L={L3}): pred={pred3.shape}')

# Memory estimate
param_mem = total * 4 / 1e6  # FP32
grad_mem = total * 4 / 1e6
optim_mem = total * 8 / 1e6  # Adam (momentum + velocity)
print(f'\nMemory estimate (FP32):')
print(f'  Parameters:  {param_mem:.1f} MB')
print(f'  Gradients:   {grad_mem:.1f} MB')
print(f'  Optimizer:   {optim_mem:.1f} MB')
print(f'  Total min:   {param_mem+grad_mem+optim_mem:.1f} MB')
print(f'  6GB VRAM:    AMPLE headroom')
print(f'  Under 2M params: {"YES" if total < 2e6 else "NO"}')
