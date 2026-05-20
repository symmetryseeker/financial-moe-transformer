"""
Differentiated expert re-initialization strategy.

Problem: Symmetric init + low SNR → 2 experts dominate, 4 collapse.
Fix:    Keep Expert 2/5 (working), re-init 1/3/4/6 with varied seeds + higher variance.
        Re-init gate with wider distribution to give new experts a fair chance.
        Noisy Top-K gating (noise_std=0.15) for gate exploration.
"""
import sys; sys.path.insert(0,'.')
import torch, torch.nn as nn, copy, pyarrow.parquet as pq, numpy as np
from model import FinancialMoETransformer
from config import model_cfg

# Load checkpoint first to get vocab dimensions
print("Loading checkpoint...")
ckpt = torch.load("checkpoints/best.pt", map_location="cpu", weights_only=False)
ckpt_n_companies = ckpt["model"]["embedding.dual_embed.company_embed.weight"].shape[0]
ckpt_n_metrics = ckpt["model"]["embedding.dual_embed.metric_embed.weight"].shape[0]
print(f"Checkpoint vocab: {ckpt_n_companies}c x {ckpt_n_metrics}m")

model = FinancialMoETransformer(
    n_companies=ckpt_n_companies, n_metrics=ckpt_n_metrics,
    d_model=128, nhead=4, num_layers=model_cfg.num_layers,
    dim_feedforward=model_cfg.dim_feedforward,
    num_experts=model_cfg.num_experts, expert_hidden=model_cfg.expert_hidden,
)

ckpt = torch.load("checkpoints/best.pt", map_location="cpu", weights_only=False)
model.load_state_dict(ckpt["model"])
print(f"Loaded: epoch {ckpt['epoch']+1}, Val IC={ckpt['best_ic']:.4f}")

# ── Save Expert 2 & 5 weights ────────────────────────────────────
expert2_state = {k: v.clone() for k, v in model.moe.experts[1].state_dict().items()}
expert5_state = {k: v.clone() for k, v in model.moe.experts[4].state_dict().items()}
gate_state_old = {k: v.clone() for k, v in model.moe.gate.state_dict().items()}

# ── Re-init Experts 1,3,4,6 with differentiated seeds ─────────────
keep_indices = {1, 4}  # Expert 2 and 5 (0-indexed: 1, 4)
reinit_indices = [0, 2, 3, 5]  # Expert 1, 3, 4, 6

for idx in reinit_indices:
    torch.manual_seed(42 + idx * 137)  # Different seed per expert
    for name, param in model.moe.experts[idx].named_parameters():
        if param.dim() >= 2:
            # 2x higher std than default to increase diversity
            nn.init.normal_(param, mean=0.0, std=0.10)
        else:
            nn.init.zeros_(param)
    print(f"  Expert {idx+1}: re-initialized (seed={42+idx*137})")

# ── Re-init Gate with higher variance ─────────────────────────────
gate = model.moe.gate
nn.init.normal_(gate.weight, mean=0.0, std=0.08)  # was 0.02; wider to give all experts fair chance
nn.init.zeros_(gate.bias)
print(f"  Gate: re-initialized (std=0.08)")

# ── Restore Expert 2 & 5 ─────────────────────────────────────────
model.moe.experts[1].load_state_dict(expert2_state)
model.moe.experts[4].load_state_dict(expert5_state)
print(f"  Expert 2 & 5: restored")

# ── Verify ────────────────────────────────────────────────────────
print(f"\n  Expert weight norms after re-init:")
for i in range(6):
    norms = [p.norm().item() for p in model.moe.experts[i].parameters() if p.dim() >= 2]
    avg_norm = sum(norms) / len(norms) if norms else 0
    tag = "(kept)" if i in keep_indices else "(re-init)"
    print(f"    Expert {i+1}: avg_norm={avg_norm:.4f} {tag}")

# ── Save ──────────────────────────────────────────────────────────
torch.save({
    "epoch": ckpt["epoch"],
    "model": model.state_dict(),
    "best_ic": ckpt["best_ic"],
    "note": "Differentiated expert re-init: E2+E5 kept, E1/3/4/6 re-init, gate re-init"
}, "checkpoints/reinit.pt")
print(f"\n  Saved: checkpoints/reinit.pt")
