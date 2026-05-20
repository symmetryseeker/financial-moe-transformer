# Financial MoE Transformer

CSI 300 excess return prediction using a Transformer + Mixture of Experts (MoE) hybrid model.

## Architecture (Phase 2 — proven)

- **Dual Embedding**: company (32d) + metric (96d) → solves vocab explosion
- **Hierarchical Encoder**: source-pooled × 4-scale temporal pyramid
- **Chunked cosFormer Attention**: O(L·d²) with chunk_size=2048
- **Sparse MoE**: 6 experts (128→64→32→1), Top-1 routing
- **EMA inference**: decay=0.999 for stable predictions

## Current State

| Metric | Value |
|--------|-------|
| Val IC | +0.053 (rising) |
| From-scratch IC | +0.039 |
| Expert CV | 1.36 |
| Params | 938K (data/processed/ dataset) |

## Key Files

```
config.py          — All hyperparameters
model/             — Architecture (embedding, transformer, moe, predictor, hierarchical)
utils/             — Dataset, losses, text encoder, LLM state generator
data/              — Data pipeline (fetch, prepare, rebuild labels)
train.py           — Training script (Phase 2 reverted — no noise/z-loss)
reinit_experts.py  — Differentiated expert re-initialization
```

## Training

```bash
# From scratch on data/processed/ dataset
python train.py --epochs 20 --batch-size 2

# Continue from checkpoint
python train.py --epochs 15 --batch-size 2 --resume checkpoints/best.pt
```

## Data

Two datasets exist:
- `data/processed/` — 316c × 333m, 15.4M rows, proven IC=+0.053
- `D:/financial_data/processed/` — 6267c × 127m, 22.2M rows, IC ceiling ~+0.027

**Vocab stability**: `_build_dual_vocab()` persists `company_vocab.json` and `metric_vocab.json` to `data/processed/cache/` to ensure checkpoint compatibility across pipeline runs.

## Key Lessons

1. **LB > 0.05 harms training** — load balance coefficient must stay low (0.01-0.03)
2. **Noise gating / z-loss have zero impact on IC** — removed in Phase 2 revert
3. **Differentiated expert init only helps when experts are collapsed** — don't reinit well-trained experts
4. **Low LR + patience pays off** — IC broke from 0.039→0.053 after epoch 9 with LR=1e-4
5. **Data pipeline stability is critical** — vocab must be persisted to reuse checkpoints
