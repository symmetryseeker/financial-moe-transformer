# Financial MoE Transformer — 架构与结果完整报告

## 1. 完整架构

### 1.1 数据流

```
data_points.parquet (15.4M rows, 316 companies × 346 metrics)
         │
         ▼
  ┌──────────────────────────────────────────────┐
  │         SlidingWindowDataset                 │
  │  窗口: 365天, 最大 4096 tokens/窗口          │
  │  每 token: value + company_id + metric_id    │
  │           + source_id + time_bins            │
  │           + day/month/dow/year_offset + tsu  │
  │  2,636 指数窗口 | 1,009,741 个股窗口         │
  └──────────────────────────────────────────────┘
         │
         ▼
  ┌──────────────────────────────────────────────┐
  │           EmbeddingLayer                     │
  │  value_proj:      Linear(1, 128)             │
  │  dual_embed:      company(32d) + metric(96d) │
  │  calendar_embed:  day(8)+month(8)+dow(8)     │
  │                   +year(8) → Linear(32,128)  │
  │  time_bucket:     bucketize(tsu) → Embed(8)  │
  │                   → Linear(8, 128)           │
  │  输出: (B, L, 128)                           │
  └──────────────────────────────────────────────┘
         │
         ▼
  ┌──────────────────────────────────────────────┐
  │       SimpleHierarchicalEncoder              │
  │  5 sources × 4 time scales = 20 groups       │
  │  每 group: mean pool → 4 scale projections   │
  │  跨源融合: 2-layer cosFormer Transformer     │
  │  输出: (B, L, 128) + (B, L) mask             │
  └──────────────────────────────────────────────┘
         │
         ▼
  ┌──────────────────────────────────────────────┐
  │          AttentionPooling                    │
  │  Learnable query (1,1,128)                   │
  │  多头注意力 pooled → (B, 128)                │
  └──────────────────────────────────────────────┘
         │
         ├─── 无财务快照: 直接送入 MoE
         │
         └─── 有财务快照: concat(128, 64) → Linear(192, 128) → MoE
                  │
                  └── FinancialEncoder: 34→128→64 (LayerNorm+GELU)
         │
         ▼
  ┌──────────────────────────────────────────────┐
  │            MoEPredictor                      │
  │  Gate: Linear(128, 6) → softmax → Top-1      │
  │  6 Experts: 128→64→32→1 (GELU+Dropout)       │
  │  输出: (B, 1) pred + load_balance_loss       │
  └──────────────────────────────────────────────┘
```

### 1.2 参数统计

| 组件 | 参数数 | 说明 |
|------|--------|------|
| EmbeddingLayer | ~300K | Value(128) + Company(316×32) + Metric(346×96) + Calendar(8+8+8+8) + TimeBucket(8) |
| HierarchicalEncoder | ~350K | cosFormer(128,4head,4layer) + 4scale_projections |
| AttentionPooling | ~16K | query(128) + key_proj(128→128) |
| MoEPredictor | ~250K | Gate(128→6) + 6×Expert(128→64→32→1) |
| FinancialEncoder | 13K | 34→128→64 |
| Fusion Projection | 8K | Linear(192, 128) |
| **总计** | **~977K** | |

### 1.3 训练配置

| 参数 | 值 |
|------|-----|
| 损失 | `MSE(1.0) + SpearmanRank(0.3) + LoadBalance(0.01)` |
| 优化器 | AdamW(lr=3e-4, wd=0.1) |
| LR 调度 | 500-step Linear warmup + CosineAnnealing |
| 早停 | patience=5 |
| EMA | decay=0.999 |
| Batch Size | 8-16 |
| Max Epochs | 15-25 |
| 训练样本 | 20,000 (从 776K 全量采样) |
| 验证样本 | 2,000 |
| 切分 | 80/10/10 时序切分 |

---

## 2. 关键实验结果

### 2.1 指数级预测

| 实验 | 配置 | Val IC | 说明 |
|------|------|--------|------|
| Phase 2 基线 | 70/15, 原始标签 | +0.039 | 从零训练天花板 |
| 继续训练 | 低 LR 微调 | **+0.053** | 从 IC=0.039 突破至 0.053 |
| 国际宏观注入 | +13 宏观变量 | +0.054 | 小幅增益 |
| WF-CV (固定模型) | 多周期评估 | -0.33 | 从零训练不稳健 |

**索引级结论**: 最佳 IC=+0.053，逆转信号择时 Sharpe 达 +0.97(2020)/+0.99(2021)。

### 2.2 截面预测

| 实验 | Val IC | 截面 Rank IC | 结论 |
|------|--------|-------------|------|
| 简单个股训练 | +0.054 | NaN | 无截面方差 |
| +CS Loss + 归一化 | +0.100 | NaN | 时序 IC 涨但截面仍 NaN |
| +target_stock_mask | -0.093 | — | 破坏训练 |
| +股票历史 token | +0.006 | — | 信号淹没 |
| +财务快照编码器 | **+0.0001** | — | **财务特征无预测力 (IC=-0.006)** |

**截面结论**: 所有尝试均未产生有效截面信号。根本原因——输入对称 + 财务特征预测力为零。

### 2.3 MoE 调参

| 实验 | 结论 |
|------|------|
| 噪声门控 (σ=0.05-0.15) | IC 零影响 |
| Z-Loss (coef=0.001) | IC 零影响 |
| LB (0.01-0.30) | LB>0.05 有害, 最优 0.01-0.03 |
| 专家差异化初始化 | 仅在专家已坍缩时有帮助 |

**MoE 结论**: 所有 MoE 调参对预测质量无影响。

### 2.4 数据切分

| 切分 | Val IC | Test IC | 结论 |
|------|--------|---------|------|
| 70/15 | +0.039 | ~+0.01 | 稳定但弱 |
| 80/10 | +0.056 | +0.009 | 略高但不显著 |
| 85/5 | +0.133 | -0.002 | **Val 太小,假信号** |

---

## 3. 当前代码库状态

### 3.1 核心文件

```
model/
├── predictor.py          ← FinancialMoETransformer (双路径: fin/nofin)
├── financial_encoder.py  ← FinancialEncoder (34→128→64, LayerNorm+GELU)
├── embedding.py          ← EmbeddingLayer (value+dual+calendar+timebucket)
├── hierarchical.py       ← SimpleHierarchicalEncoder (source×scale pooling)
├── moe.py                ← MoEPredictor (6 experts, Top-1 gating)
└── transformer.py        ← cosFormer chunked linear attention

utils/
├── dataset.py            ← SlidingWindowDataset (指数级)
├── dataset_stock.py      ← StockSlidingWindowDataset (个股级, 含财务快照)
└── losses.py             ← CombinedLoss (MSE+SpearmanRank+LB, 无CS Loss)

data/
├── prepare_data.py       ← 完整数据管线
├── build_stock_labels.py ← 个股标签生成 (1M+ 行)
├── build_financial_snapshots.py ← 财务快照表 (622K 行, 34 特征)
├── inject_macro.py       ← 国际宏观注入 (13 变量)
└── add_stock_history.py  ← 股票历史特征 (699K 行, 5 特征)

checkpoints/
├── best.pt               ← 原始 Phase 2 (IC=0.063, 旧数据)
├── best_phase2_cont.pt   ← 最佳指数模型 (IC=0.053)
├── best_cs.pt            ← 最佳 CS 训练 (IC=0.100, 截面 NaN)
└── best_fin_encoder.pt   ← 财务编码器 (IC=0.0001, 实验失败)
```

### 3.2 数据集

| 文件 | 行数 | 列 |
|------|------|-----|
| data/processed/data_points.parquet | 15.4M | [datetime, source, variable, value, value_raw, time_since_update] |
| data/processed/labels.parquet | 2,514 | [datetime, label] (指数级) |
| data/processed/labels_stock.parquet | 1,009,741 | [datetime, stock_code, label] (个股级) |
| data/processed/financial_snapshots.parquet | 622K | [datetime, stock_code, feat_0...feat_33] |
| D:/financial_data/market/csi300_stocks_daily.csv | 701K | 股票日线 (298 只) |

---

## 4. 失败路径总结

| 序号 | 尝试 | 失败原因 | 代价 |
|------|------|---------|------|
| 1 | CS Loss + Z-Score 归一化 | 训练时伪造方差，推理时塌缩 | ~2 天 |
| 2 | target_stock_mask | mask 信号淹没在 8192 tokens 中 | ~1 天 |
| 3 | 股票历史 token (300/窗口) | token 过多稀释主信号 | ~1 天 |
| 4 | FinancialEncoder (34 字段) | 财务特征原始 IC≈0，无可学习的信号 | ~0.5 天 |

**核心教训**: 
- 在不到 20K 训练样本上，无法学到个股间的截面差异
- 输入对称问题是架构层面的，不能用损失函数弥补
- 财务特征需要更好的工程处理：使用 feature 的一阶差分（变化率）、或使用行业相对值、或使用非线性组合/MoE 特征交叉，而不是绝对水平本身
- 有效的截面信号需要不对称输入——每只股票必须看到不同的信息。(下一步方向：可以借助你的学术文献 MCP 获取财务特征工程的已有方法，获取各个数据库的截面因子研究，并重点下载以下内容：财务特征工程（Financial Feature Engineering）：关注 RESSET/CSMAR 数据库中关于截面因子（Cross-Sectional Factors）和财务特征工程的论文。特别是那些讨论 财务比率的变化率（如 ΔROE、Δleverage）、行业相对值（Industry-Relative Ratios），或非线性因子（Non-Linear Factors）的文章。可以从 arXiv 搜索 'financial feature engineering cross-sectional factors' 或 'industry-relative financial ratios stock prediction'。时间序列-截面融合方法（Time-Series & Cross-Sectional Fusion）：寻找研究如何将 时序预测（Time-Series Predictions）与 截面特征（Cross-Sectional Features） 融合的论文。特别是那些在 Transformer 或 MoE 架构下进行 多模态/多任务学习 的论文，看它们如何处理 时序信号与截面信号的互补关系。跨市场/跨资产通用因子：在 EBSCO、JCR 或 Web of Science 中搜索 'multi-market cross-asset common factors stock prediction'，了解国际市场（VIX、DXY、US10Y）与中国市场的联动规律。)

---

## 5. 有效成果

| 成果 | 详情 |
|------|------|
| 指数择时策略 | Sharpe +0.97 (2020), +0.47 (2024-25) |
| GPU 训练 | 5-7 min/epoch, 10x 加速 |
| 数据管线 | 1M+ 个股标签, 622K 财务快照 |
| 多周期评估 | 固定模型在 5 个市场制度下的表现 |
| Word 报告 | 10 章节, 完整回测记录 |
| GitHub | [symmetryseeker/financial-moe-transformer](https://github.com/symmetryseeker/financial-moe-transformer) |
