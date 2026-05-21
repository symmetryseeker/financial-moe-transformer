"""V3 model parameter audit + forward pass test."""
import sys; sys.path.insert(0, '.')
import torch
from model import FinancialMoETransformer
from config import model_cfg

print("Building V3 model...")
model = FinancialMoETransformer(
    vocab_size=model_cfg.vocab_size,
    d_model=model_cfg.d_model, nhead=model_cfg.nhead,
    num_layers=model_cfg.num_layers,
    dim_feedforward=model_cfg.dim_feedforward,
    attn_type=model_cfg.attn_type,
    attn_chunk_size=model_cfg.attn_chunk_size,
    num_experts=model_cfg.num_experts,
    expert_hidden=model_cfg.expert_hidden,
    top_k=model_cfg.top_k,
    max_seq_len=model_cfg.max_seq_len,
)

total = sum(p.numel() for p in model.parameters())
print(f"\nTotal params: {total:,}")

# Breakdown
for name, mod in model.named_children():
    n = sum(p.numel() for p in mod.parameters())
    print(f"  {name:<20s}: {n:>10,}")
    for sub, submod in mod.named_children():
        sn = sum(p.numel() for p in submod.parameters())
        if sn > 500:
            print(f"    {sub:<18s}: {sn:>10,}")

print(f"\nVRAM (FP32):     {total*4/1e6:.1f} MB params")
print(f"VRAM (+grad+opt): {total*16/1e6:.1f} MB total")
print(f"Under 2M: {'YES' if total < 2e6 else 'NO'}")

# Forward pass with random data
print("\nForward pass test...")
B, L = 2, 512
batch = {
    "values": torch.randn(B, L, 1),
    "var_ids": torch.randint(1, 800, (B, L)),
    "day": torch.randint(1, 31, (B, L)),
    "month": torch.randint(1, 12, (B, L)),
    "dow": torch.randint(0, 6, (B, L)),
    "year_offset": torch.randint(0, 10, (B, L)),
    "time_since_update": torch.rand(B, L) * 30,
    "mask": torch.ones(B, L, dtype=torch.bool),
    "source_ids": torch.randint(0, 4, (B, L)),
    "time_bins": torch.randint(0, 4, (B, L)),
}

model.eval()
with torch.no_grad():
    pred, lb = model(**batch)
print(f"  pred shape: {pred.shape}, lb_loss: {lb.item():.4f}")

# Backward pass
model.train()
pred, lb = model(**batch)
loss = torch.nn.functional.mse_loss(pred.squeeze(-1), torch.randn(B)) + 0.01 * lb
loss.backward()
print(f"  Gradients OK, loss={loss.item():.4f}")

print("\nV3 VERIFIED")
