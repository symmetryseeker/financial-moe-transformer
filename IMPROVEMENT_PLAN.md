# V2 → V3 完整改进方案

> 基线: Val IC = +0.016 | 专家退化 2/6 | 110万参数 | 63天预测周期

---

## 一、立即修复（预计 30 分钟，预期 IC +0.005~0.010）

### 1.1 负载均衡系数提升

**问题**: Expert 1 和 6 退化（0%使用），CV=1.23，负载均衡损失太弱。

**修复** (`config.py`):
```python
LOAD_BALANCE_COEF = 0.05   # 0.01 → 0.05
```

**原理**: 当前系数 0.01 在总损失中被 MSE（系数 1.0）完全淹没。升至 0.05 后约占总损失 2-3%，足够推动专家激活但不扭曲主任务。

### 1.2 方向损失权重提升

**问题**: Direction 损失稳定在 0.69，从未改善——说明它没有提供学习信号，标签是 50/50 方向平衡的。

**修复** (`config.py`):
```python
LOSS_DIRECTION_COEF = 0.0   # 0.1 → 0（关闭）
# 同时增加排序损失补偿:
LOSS_RANK_COEF = 0.5        # 0.3 → 0.5
```

**原理**: 标签方向 49/51 均衡，模型学不到方向信号。将算力集中在排序损失上。

### 1.3 去掉未使用的 expert_layers 配置

**问题**: `config.py` 定义了 `EXPERT_LAYERS=3` 但 moe.py 硬编码了深度，参数未生效。

**修复**: 从 `ModelConfig` 和 `TrainConfig` 中删除 `expert_layers` 字段，或将其传入 `MoEPredictor` 构造函数。

---

## 二、架构微调（预计 1 小时，预期 IC +0.005~0.015）

### 2.1 MoE Top-2 路由

**问题**: Top-1 路由导致 2 个专家完全退化。Top-2 让每个样本激活 2 个专家，样本分配更均匀。

**修复** (`config.py` + `model/moe.py`):
```python
TOP_K = 2   # 1 → 2
```

**原理**: 每个专家获得的样本量翻倍，Expert 1/6 能分配到辅助路由的样本。推理时仍可选 Top-1。

### 2.2 专家初始化噪声

**问题**: 6 个专家初始权重相同（对称初始化），导致部分专家"输在起跑线"。

**修复** (`model/moe.py`):
```python
# 在 MoEPredictor.__init__ 中，专家使用不同的随机种子
for i, expert in enumerate(self.experts):
    torch.manual_seed(42 + i)
    for p in expert.parameters():
        nn.init.normal_(p, mean=0, std=0.02 * (0.5 + i * 0.2))
```

### 2.3 添加预测不确定性输出

**问题**: 模型只输出点预测，无法量化置信度。Expert 2 的 IC=0.34 但只用 12%——说明高置信度样本确实存在。

**修复** (`model/moe.py`): 每个专家同时输出预测值和置信度:
```python
# Expert 输出改为 (pred, log_var)
# 训练时用 Gaussian NLL 损失加权
```

**简化版**: 用专家间的预测方差作为不确定性代理——无需改架构。

---

## 三、数据增强（预计 2 小时，预期 IC +0.005~0.010）

### 3.1 多窗口长度采样

**问题**: 固定 365 天窗口可能不是最优。不同市场状态下最佳回溯期不同。

**修复** (`utils/dataset.py`):
```python
# 训练时随机选择窗口长度
WINDOW_DAYS = random.choice([126, 252, 365, 504])  # 6月/1年/1.5年/2年
```

### 3.2 标签截面标准化

**问题**: 63 天超额收益的波动率随时间变化（高波动期的信号更强），异方差降低了排序损失的效果。

**修复** (`data/prepare_data.py`):
```python
# 用过去 252 天的收益波动率做标准化
label_vol = labels['label'].rolling(252).std()
labels['label_normalized'] = labels['label'] / label_vol
```

### 3.3 补齐缺失数据

| 数据 | 方法 | 时间 |
|------|------|------|
| 国际宏观 | yfinance 已采集 19 个品种 | ✅ 已完成 |
| DeepSeek 市场状态 | `utils/deepseek_state.py` 已就绪 | 10 分钟 |
| 文本嵌入 | AKShare 新闻 → BGE 编码 | 需要代理 |

---

## 四、训练策略优化（预计 30 分钟，预期 IC +0.002~0.005）

### 4.1 学习率 Warmup 延长

**问题**: 当前 warmup=500 步，对于 920 batch/epoch 来说约半个 epoch。模型在初期可能震荡。

**修复** (`train.py`):
```python
WARMUP_STEPS = min(1000, total_steps // 3)  # 500 → 延长
```

### 4.2 验证集扩大

**问题**: 验证集仅 394 个样本，IC 估计的置信区间约 ±0.10。波动大。

**修复**: 将 test 集的一部分并入 val:
```python
TRAIN_FRAC = 0.6   # 0.7 → 0.6
VAL_FRAC = 0.25    # 0.15 → 0.25  
TEST_FRAC = 0.15
```

### 4.3 保存多个最佳检查点

**问题**: 只保存一个 best.pt，IC 波动时可能错过稍差但更鲁棒的模型。

**修复** (`train.py`):
```python
# 保存 top-3 检查点
if len(best_models) < 3 or val_ic > min(best_models):
    save_checkpoint()
```

---

## 五、分步执行顺序

```
Phase 1 (立即执行，30分钟):
  ├── LOAD_BALANCE_COEF: 0.01 → 0.05
  ├── LOSS_DIRECTION_COEF: 0.1 → 0.0
  ├── LOSS_RANK_COEF: 0.3 → 0.5
  ├── Top-K: 1 → 2
  └── 重新训练 → 目标 IC > 0.02

Phase 2 (本周，2小时):
  ├── 标签波动率标准化
  ├── 多窗口长度采样
  ├── 验证集扩大
  └── 重新训练 → 目标 IC > 0.03

Phase 3 (下周，4小时):
  ├── DeepSeek API 市场状态生成
  ├── 文本嵌入 (AKShare + BGE)
  ├── 专家初始化改进
  └── 重新训练 → 目标 IC > 0.04
```

---

## 六、预期提升路径

```
当前 IC:  +0.016
         │
Phase 1: +0.025  (+56%)  ← 专家激活 + 损失优化
         │
Phase 2: +0.035  (+40%)  ← 标签优化 + 数据增强
         │
Phase 3: +0.045  (+29%)  ← 文本 + 市场状态
         │
         ↓
目标 IC: ~0.04-0.05（金融 ML 实用门槛）
```
