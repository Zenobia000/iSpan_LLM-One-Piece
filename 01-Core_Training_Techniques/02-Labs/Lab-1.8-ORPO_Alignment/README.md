# Lab-1.8: ORPO 對齊優化
## Odds Ratio Preference Optimization (ORPO)

**實驗室類型**: 模型對齊技術
**難度等級**: ⭐⭐⭐⭐⭐ (高級)
**預估時間**: 4-6小時
**適用GPU**: 16GB+ VRAM

---

## 📚 實驗室概述

ORPO (Odds Ratio Preference Optimization) 是比 DPO 更先進的對齊技術，通過單階段訓練同時完成 instruction tuning 和偏好對齊，無需預先訓練 SFT 模型，大幅簡化對齊流程並提升效率。

### 學習目標

完成本實驗室後，您將能夠：
- ✅ 理解 ORPO 相比 DPO 的創新點
- ✅ 掌握 Odds Ratio 損失函數原理
- ✅ 實現單階段對齊訓練
- ✅ 對比 SFT+DPO vs ORPO 的效果
- ✅ 優化 ORPO 訓練超參數

---

## 🎯 ORPO 核心創新

### DPO 的局限

**DPO 流程**:
```
Phase 1: SFT (Supervised Fine-Tuning)
  → 訓練基礎能力
  → 需要 instruction dataset

Phase 2: DPO (Direct Preference Optimization)
  → 對齊人類偏好
  → 需要 preference dataset

問題:
  - 需要兩個階段
  - 需要兩種數據集
  - SFT 和 DPO 目標可能衝突
```

### ORPO 的解決方案

**ORPO 單階段訓練**:
```
Single Phase: ORPO
  → 同時學習 instruction following + preference alignment
  → 只需要 preference dataset (包含 instruction)

優勢:
  ✅ 單階段訓練 (無需 SFT)
  ✅ 訓練時間減少 50%
  ✅ 避免 SFT-DPO 目標衝突
  ✅ 統一優化目標
```

### ORPO 損失函數

**組合損失**:
```
L_ORPO = L_SFT + λ × L_OR

其中:
  L_SFT = 標準語言模型損失 (對 chosen responses)
  L_OR = Odds Ratio 偏好損失
  λ = 平衡權重 (通常 0.1-1.0)
```

**Odds Ratio 損失**:
```python
# Odds Ratio: 衡量 chosen vs rejected 的相對機率
odds_ratio = (P(chosen) / (1 - P(chosen))) / (P(rejected) / (1 - P(rejected)))

# Log Odds Ratio
log_odds = log(P(chosen) / (1 - P(chosen))) - log(P(rejected) / (1 - P(rejected)))
         = log(P(chosen)) - log(P(rejected)) - log(1 - P(chosen)) + log(1 - P(rejected))

# ORPO 損失 (最大化 log odds)
L_OR = -log_sigmoid(log_odds)
```

### ORPO vs DPO 對比

| 特性 | DPO | ORPO | 優勢 |
|------|-----|------|------|
| **訓練階段** | 2階段 (SFT + DPO) | 1階段 | ORPO ⬆ |
| **所需數據** | Instruction + Preference | Preference (含 instruction) | ORPO ⬆ |
| **參考模型** | 需要 (SFT model) | 不需要 | ORPO ⬆ |
| **訓練時間** | 100% | 50-60% | ORPO ⬆ |
| **記憶體占用** | 2x (policy + ref) | 1x | ORPO ⬆ |
| **對齊效果** | 強 | 更強 | ORPO ⬆ |
| **實現複雜度** | 中等 | 簡單 | ORPO ⬆ |

---

## 📂 實驗室結構

```
Lab-1.8-ORPO_Alignment/
├── README.md                         # 本文檔
├── 01-Setup.ipynb                   # 環境設置與數據準備
├── 02-ORPO_Training.ipynb           # ORPO 單階段訓練
├── 03-Compare_with_DPO.ipynb        # vs DPO 對比實驗
└── 04-Production_Deploy.ipynb       # 生產部署指南
```

---

## 🔧 技術原理詳解

### Odds Ratio 數學原理

**定義**:
```
Odds(事件) = P(事件發生) / P(事件不發生)
           = P / (1 - P)

Odds Ratio = Odds(chosen) / Odds(rejected)
```

**在 ORPO 中的應用**:
```python
# 計算每個 response 的 odds
def compute_odds(logits, labels):
    """計算序列的 odds"""
    # P(sequence) = ∏ P(token_i | context)
    log_probs = F.log_softmax(logits, dim=-1)
    token_log_probs = torch.gather(log_probs, -1, labels.unsqueeze(-1)).squeeze(-1)

    # Log P(sequence)
    log_p = token_log_probs.sum(dim=-1)

    # Log odds = log(P / (1-P))
    # 近似: log_odds ≈ log_p (when P << 1)
    log_odds = log_p - torch.log1p(-torch.exp(log_p))

    return log_odds

# ORPO 偏好損失
log_odds_chosen = compute_odds(logits_chosen, chosen_labels)
log_odds_rejected = compute_odds(logits_rejected, rejected_labels)

L_OR = -F.logsigmoid(log_odds_chosen - log_odds_rejected).mean()
```

### 完整 ORPO 算法

```python
def orpo_loss(model, batch, lambda_or=0.5):
    """
    ORPO 損失函數

    Args:
        model: 訓練中的模型
        batch: {'prompt', 'chosen', 'rejected'}
        lambda_or: OR 損失權重

    Returns:
        total_loss, metrics
    """
    # 1. SFT 損失 (標準 LM loss on chosen)
    prompt_chosen = torch.cat([batch['prompt'], batch['chosen']], dim=1)
    outputs_chosen = model(prompt_chosen, labels=prompt_chosen)
    L_SFT = outputs_chosen.loss

    # 2. Odds Ratio 損失
    # Chosen
    chosen_logits = outputs_chosen.logits[:, batch['prompt'].size(1)-1:-1, :]
    chosen_labels = batch['chosen']
    log_odds_chosen = compute_odds(chosen_logits, chosen_labels)

    # Rejected
    prompt_rejected = torch.cat([batch['prompt'], batch['rejected']], dim=1)
    outputs_rejected = model(prompt_rejected)
    rejected_logits = outputs_rejected.logits[:, batch['prompt'].size(1)-1:-1, :]
    rejected_labels = batch['rejected']
    log_odds_rejected = compute_odds(rejected_logits, rejected_labels)

    # OR 損失
    L_OR = -F.logsigmoid(log_odds_chosen - log_odds_rejected).mean()

    # 3. 總損失
    total_loss = L_SFT + lambda_or * L_OR

    metrics = {
        'loss': total_loss.item(),
        'sft_loss': L_SFT.item(),
        'or_loss': L_OR.item(),
        'log_odds_margin': (log_odds_chosen - log_odds_rejected).mean().item()
    }

    return total_loss, metrics
```

---

## 📊 實驗內容詳解

### Notebook 1: 環境設置 (01-Setup.ipynb)
**時間**: 30-45分鐘

**內容**:
- ORPO 環境驗證
- 偏好數據集準備
- 數據預處理與統計

### Notebook 2: ORPO 訓練 (02-ORPO_Training.ipynb)
**時間**: 90-120分鐘

**內容**:
- ORPO 損失函數實現
- 單階段訓練
- 訓練指標監控
- 模型檢查點保存

**關鍵代碼**:
```python
from trl import ORPOTrainer, ORPOConfig

config = ORPOConfig(
    beta=0.1,              # OR 溫度
    lambda_or=0.5,         # OR 損失權重
    learning_rate=5e-6,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=16,
    num_train_epochs=1
)

trainer = ORPOTrainer(
    model=model,
    args=config,
    train_dataset=preference_dataset,
    tokenizer=tokenizer
)

trainer.train()
```

### Notebook 3: vs DPO 對比 (03-Compare_with_DPO.ipynb)
**時間**: 60-75分鐘

**對比維度**:
- 訓練時間
- 記憶體占用
- 對齊效果
- 模型質量

### Notebook 4: 生產部署 (04-Production_Deploy.ipynb)
**時間**: 45-60分鐘

**內容**:
- 模型導出與量化
- 推理優化
- 部署最佳實踐
- 持續改進策略

---

## 📈 性能預期

### ORPO vs DPO vs SFT

| 方法 | 訓練階段 | GPU 時 | Win Rate | Helpfulness | 相對成本 |
|------|---------|-------|---------|-------------|---------|
| **SFT** | 1 | 20 | 50% | 3.5/5 | 20% |
| **SFT+DPO** | 2 | 40 | 68% | 4.1/5 | 100% |
| **ORPO** | 1 | 25 | 70% | 4.2/5 | 62% |

**ORPO 優勢**:
- ✅ 效果最佳
- ✅ 成本最低 (單階段)
- ✅ 訓練最簡單

---

## 💡 最佳實踐

### 超參數建議

```python
# 推薦配置
orpo_config = {
    'beta': 0.1,           # OR 溫度 (0.05-0.2)
    'lambda_or': 0.5,      # OR 損失權重 (0.1-1.0)
    'learning_rate': 5e-6, # 較小學習率
    'warmup_ratio': 0.1,
    'max_length': 512
}
```

### 訓練技巧

1. **Lambda 調優**: 控制 SFT vs OR 平衡
2. **監控 odds margin**: 應該持續增長
3. **早停策略**: 避免過擬合

---

## 🎓 學習檢查清單

- [ ] 理解 Odds Ratio 概念
- [ ] 實現 ORPO 損失函數
- [ ] 訓練 ORPO 模型
- [ ] 對比 ORPO vs DPO 效果

---

## 🚀 下一步

完成本實驗室後:
- 掌握最先進的對齊技術
- 可部署生產級對齊模型
- 了解對齊技術的前沿發展

---

**實驗室狀態**: 🔄 開發中
**最後更新**: 2025-10-08

**相關論文**:
- [ORPO: Monolithic Preference Optimization](https://arxiv.org/abs/2403.07691) - 2024
- [DPO: Direct Preference Optimization](https://arxiv.org/abs/2305.18290)
