# Financial MoE Transformer · 金融MoE Transformer

CSI 300 index prediction via Transformer + Mixture of Experts.  
基于 Transformer + 混合专家模型的沪深300指数预测。

[![GPU](https://img.shields.io/badge/GPU-GTX%201060%206GB-green)]()
[![PyTorch](https://img.shields.io/badge/PyTorch-2.5.1%2Bcu121-orange)]()
[![IC](https://img.shields.io/badge/Val%20IC-0.100-blue)]()

---

## Architecture · 架构

| Component 组件 | Description 描述 |
|---|---|
| Dual Embedding 双嵌入 | company(32d) + metric(96d) → 128d token |
| Hierarchical Encoder 层次编码器 | 5 sources × 4 time scales, cross-source fusion |
| cosFormer Attention 线性注意力 | chunk=2048, O(L·d²), supports 8192 tokens |
| Sparse MoE 稀疏专家 | 6 experts, Top-1 routing, 938K params |

## Key Results · 核心结果

| Metric 指标 | Value 数值 |
|---|---|
| Best Val IC 最佳验证IC | **+0.100** |
| Market Timing Sharpe 择时夏普 | **+0.97** (2020) / **+0.99** (2021) |
| GPU Speed 速度 | 5-7 min/epoch (10× vs CPU) |
| Training Data 训练数据 | 1M+ stock labels, 15-22M data points |

## Quick Start · 快速开始

```bash
# Train 训练
python train.py --epochs 20 --batch-size 4

# Market timing backtest 择时回测
python backtest_phase1_final.py

# Multi-period evaluation 多周期评估
python multi_period_eval.py
```

## Project Structure · 项目结构

```
model/         — Transformer, MoE, Embedding, Hierarchical Encoder
utils/         — Dataset, Losses, Text Encoder
data/          — Data pipeline, Label generation, Macro injection
reports/       — Backtest reports, Charts, Word report
checkpoints/   — Trained model weights (.pt files)
```

## Key Lessons · 关键经验

1. **Data > Architecture** 数据优于架构 — 90%+ IC improvement from data quality fixes
2. **MoE tuning is marginal** MoE调参收益有限 — noise/z-loss/LB have near-zero impact
3. **CS prediction needs asymmetric inputs** 截面预测需非对称输入 — stock-specific features required
4. **Vocab persistence is critical** 词表持久化至关重要 — save `company_vocab.json` across runs
5. **Prefixed model > WF-CV retraining** 预训练模型优于逐折重训 — use full history, deploy forward

## License

MIT
