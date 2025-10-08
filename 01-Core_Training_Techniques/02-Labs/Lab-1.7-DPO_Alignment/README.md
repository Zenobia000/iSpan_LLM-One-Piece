# Lab-1.7: DPO 直接偏好優化
## Direct Preference Optimization (DPO)

**實驗室類型**: 模型對齊技術
**難度等級**: ⭐⭐⭐⭐⭐ (高級)
**預估時間**: 5-7小時
**適用GPU**: 16GB+ VRAM

---

## 📚 實驗室概述

Direct Preference Optimization (DPO) 是一種創新的 LLM 對齊技術，無需複雜的強化學習 (RLHF)，直接從偏好數據中學習，使模型輸出更符合人類偏好。本實驗室將深入探索 DPO 的原理、實現與實際應用。

### 學習目標

完成本實驗室後，您將能夠：
- ✅ 理解 RLHF 的局限性與 DPO 的優勢
- ✅ 掌握 DPO 的數學原理與算法
- ✅ 準備偏好數據集 (preference pairs)
- ✅ 實現完整的 DPO 訓練流程
- ✅ 評估對齊效果與模型質量
- ✅ 對比 SFT vs DPO 的差異

---

## 🎯 核心技術概覽

### 為什麼需要 DPO?

**傳統 RLHF 的問題**:
```
RLHF 流程 (Reinforcement Learning from Human Feedback):
  Phase 1: Supervised Fine-Tuning (SFT)
  Phase 2: Reward Model Training
  Phase 3: PPO Training (強化學習)

問題:
  ❌ 流程複雜 (3個階段)
  ❌ PPO 訓練不穩定
  ❌ 需要維護多個模型 (policy, value, reward)
  ❌ 超參數調優困難
  ❌ 計算成本高昂
```

**DPO 的創新**:
```
DPO 流程:
  Phase 1: Supervised Fine-Tuning (SFT) - 可選
  Phase 2: DPO Training (直接優化)

優勢:
  ✅ 流程簡單 (1-2階段)
  ✅ 訓練穩定
  ✅ 只需要 policy model
  ✅ 容易調優
  ✅ 成本降低 50-70%
```

### DPO 核心原理

#### 傳統 RLHF 目標函數

```
RLHF 目標: 最大化獎勵, 同時保持與參考模型接近

max E[r(x, y)] - β KL(π_θ || π_ref)

其中:
  r(x, y): 獎勵模型評分
  π_θ: 當前策略 (policy model)
  π_ref: 參考模型 (SFT model)
  β: KL 散度權重
```

#### DPO 核心洞察

**關鍵發現**: 可以直接從偏好數據優化，無需顯式獎勵模型！

```
偏好數據格式:
  (x, y_w, y_l)
  x: prompt (輸入)
  y_w: preferred response (贏的回答)
  y_l: rejected response (輸的回答)

DPO 損失函數:
  L_DPO = -log σ(β log(π_θ(y_w|x)/π_ref(y_w|x))
                  - β log(π_θ(y_l|x)/π_ref(y_l|x)))

其中:
  σ: sigmoid 函數
  β: 溫度參數 (通常 0.1-0.5)
  π_θ: 訓練中的模型
  π_ref: 參考模型 (通常是 SFT 模型)
```

**直觀理解**:
- 增加 preferred response 的機率
- 降低 rejected response 的機率
- 通過 KL 散度保持與參考模型接近

### DPO vs RLHF 對比

| 特性 | RLHF (PPO) | DPO | 優勢 |
|------|-----------|-----|------|
| **訓練階段** | 3階段 | 1-2階段 | DPO ⬆ |
| **所需模型** | 4個 (policy, value, reward, ref) | 2個 (policy, ref) | DPO ⬆ |
| **訓練穩定性** | 不穩定 (RL) | 穩定 (監督學習) | DPO ⬆ |
| **超參數調優** | 困難 (>10個) | 簡單 (~3個) | DPO ⬆ |
| **計算成本** | 高 | 中等 | DPO ⬆ |
| **實現複雜度** | 複雜 | 簡單 | DPO ⬆ |
| **效果** | 強 | 相當 | 相近 |

---

## 📂 實驗室結構

```
Lab-1.7-DPO_Alignment/
├── README.md                         # 本文檔
├── 01-Setup_and_Data.ipynb          # 環境與偏好數據準備
├── 02-SFT_Baseline.ipynb            # SFT 基準模型訓練
├── 03-DPO_Training.ipynb            # DPO 對齊訓練
└── 04-Evaluation_and_Compare.ipynb  # 對齊效果評估
```

---

## 📊 實驗內容詳解

### Notebook 1: 環境與偏好數據準備 (01-Setup_and_Data.ipynb)
**時間**: 60-90分鐘

#### 實驗目標
- 理解偏好數據格式
- 準備 DPO 訓練數據集
- 實現數據載入器
- 探索性數據分析

#### 數據格式

**偏好對 (Preference Pair)**:
```json
{
  "prompt": "Explain what is machine learning",
  "chosen": "Machine learning is a subset of AI that enables systems to learn from data...",
  "rejected": "ML is when computers learn stuff."
}
```

**數據來源**:
- **Anthropic HH-RLHF**: 人類偏好對話數據
- **OpenAI Summarization**: 摘要偏好數據
- **Stanford SHP**: StackExchange 偏好數據
- **自定義數據**: 使用 GPT-4 生成偏好對

#### 實驗內容

1. **數據集載入**
   ```python
   from datasets import load_dataset

   # 載入 Anthropic HH-RLHF 數據集
   dataset = load_dataset("Anthropic/hh-rlhf")

   # 查看數據格式
   print(dataset['train'][0])
   # {
   #   'chosen': '...',
   #   'rejected': '...'
   # }
   ```

2. **數據預處理**
   ```python
   def preprocess_preference_data(example, tokenizer):
       """處理偏好數據"""
       # 提取 prompt 和 responses
       prompt = extract_prompt(example['chosen'])
       chosen = extract_response(example['chosen'])
       rejected = extract_response(example['rejected'])

       # Tokenize
       prompt_tokens = tokenizer(prompt)
       chosen_tokens = tokenizer(chosen)
       rejected_tokens = tokenizer(rejected)

       return {
           'prompt': prompt_tokens,
           'chosen': chosen_tokens,
           'rejected': rejected_tokens
       }
   ```

3. **數據統計分析**
   - 偏好對數量
   - 長度分布
   - 質量評估

4. **DataLoader 實現**
   ```python
   class PreferenceDataCollator:
       """DPO 數據批次收集器"""
       def __call__(self, features):
           # 批次處理 prompt, chosen, rejected
           batch = {
               'prompt_ids': pad_sequence([f['prompt'] for f in features]),
               'chosen_ids': pad_sequence([f['chosen'] for f in features]),
               'rejected_ids': pad_sequence([f['rejected'] for f in features])
           }
           return batch
   ```

---

### Notebook 2: SFT 基準模型訓練 (02-SFT_Baseline.ipynb)
**時間**: 60-90分鐘

#### 實驗目標
- 訓練 SFT (Supervised Fine-Tuning) 基準模型
- 建立參考模型 (π_ref)
- 評估 SFT 模型質量
- 為 DPO 準備初始模型

#### SFT 訓練流程

```python
# Phase 1: SFT - 在 instruction 數據上微調
model = AutoModelForCausalLM.from_pretrained("gpt2")

# 訓練數據: (instruction, response) 對
sft_dataset = [
    {"instruction": "...", "response": "..."},
    ...
]

# 標準監督學習
for batch in sft_dataloader:
    loss = model(**batch).loss
    loss.backward()
    optimizer.step()

# 得到 SFT 模型 (作為 DPO 的參考模型和初始模型)
```

#### 實驗內容

1. **基礎模型選擇**
   - GPT-2 (124M) - 快速實驗
   - Llama-2-7B - 生產級別
   - Mistral-7B - SOTA 性能

2. **SFT 數據準備**
   ```python
   # 使用 Alpaca 或類似 instruction dataset
   from datasets import load_dataset

   sft_data = load_dataset("tatsu-lab/alpaca")
   # 或使用 HH-RLHF 的 chosen responses
   ```

3. **SFT 訓練**
   - 訓練 2-3 epochs
   - 監控 loss 和 perplexity
   - 保存最佳 checkpoint

4. **質量評估**
   - 生成樣本評估
   - Perplexity 測試
   - 人工評估 (可選)

---

### Notebook 3: DPO 對齊訓練 (03-DPO_Training.ipynb)
**時間**: 90-120分鐘

#### 實驗目標
- 實現 DPO 損失函數
- 訓練 DPO 對齊模型
- 監控訓練過程指標
- 分析對齊效果

#### DPO 損失函數實現

```python
import torch.nn.functional as F

def dpo_loss(policy_model, ref_model, batch, beta=0.1):
    """
    DPO 損失函數

    Args:
        policy_model: 訓練中的策略模型
        ref_model: 參考模型 (SFT, 凍結)
        batch: {'prompt', 'chosen', 'rejected'}
        beta: 溫度參數

    Returns:
        loss, metrics
    """
    # 計算 log probabilities
    with torch.no_grad():
        # 參考模型 log probs
        ref_chosen_logps = get_log_probs(ref_model, batch['prompt'], batch['chosen'])
        ref_rejected_logps = get_log_probs(ref_model, batch['prompt'], batch['rejected'])

    # 策略模型 log probs
    policy_chosen_logps = get_log_probs(policy_model, batch['prompt'], batch['chosen'])
    policy_rejected_logps = get_log_probs(policy_model, batch['prompt'], batch['rejected'])

    # DPO loss
    chosen_rewards = beta * (policy_chosen_logps - ref_chosen_logps)
    rejected_rewards = beta * (policy_rejected_logps - ref_rejected_logps)

    # Bradley-Terry 模型
    loss = -F.logsigmoid(chosen_rewards - rejected_rewards).mean()

    # 隱式獎勵
    implicit_rewards_chosen = (policy_chosen_logps - ref_chosen_logps).detach()
    implicit_rewards_rejected = (policy_rejected_logps - ref_rejected_logps).detach()

    metrics = {
        'loss': loss.item(),
        'rewards_chosen': implicit_rewards_chosen.mean().item(),
        'rewards_rejected': implicit_rewards_rejected.mean().item(),
        'reward_margin': (implicit_rewards_chosen - implicit_rewards_rejected).mean().item()
    }

    return loss, metrics


def get_log_probs(model, prompt_ids, response_ids):
    """計算序列的 log probability"""
    # 組合 prompt + response
    input_ids = torch.cat([prompt_ids, response_ids], dim=1)

    # 前向傳播
    outputs = model(input_ids)
    logits = outputs.logits

    # 計算 log probs (只計算 response 部分)
    prompt_len = prompt_ids.size(1)
    response_logits = logits[:, prompt_len-1:-1, :]  # 對應 response tokens
    response_labels = response_ids

    # Log softmax
    log_probs = F.log_softmax(response_logits, dim=-1)

    # 收集對應 token 的 log prob
    gathered_log_probs = torch.gather(
        log_probs,
        dim=2,
        index=response_labels.unsqueeze(2)
    ).squeeze(2)

    # 平均 (或求和)
    return gathered_log_probs.sum(dim=1)
```

#### 實驗內容

1. **DPO Trainer 實現**
   ```python
   class DPOTrainer:
       def __init__(self, policy_model, ref_model, beta=0.1):
           self.policy_model = policy_model
           self.ref_model = ref_model
           self.beta = beta

           # 凍結參考模型
           for param in self.ref_model.parameters():
               param.requires_grad = False

       def train_step(self, batch):
           loss, metrics = dpo_loss(
               self.policy_model,
               self.ref_model,
               batch,
               self.beta
           )
           return loss, metrics
   ```

2. **訓練循環**
   - 訓練 1-3 epochs
   - 監控 reward margin (chosen vs rejected)
   - 早停機制

3. **關鍵指標監控**
   - **Reward Margin**: chosen 和 rejected 的獎勵差距
   - **KL Divergence**: 與參考模型的距離
   - **Accuracy**: 模型是否偏好 chosen over rejected

4. **超參數調優**
   - **β (beta)**: 0.1-0.5 (越大越保守)
   - **Learning Rate**: 1e-6 to 5e-6
   - **Epochs**: 1-3

---

### Notebook 4: 對齊效果評估 (04-Evaluation_and_Compare.ipynb)
**時間**: 45-60分鐘

#### 實驗目標
- 評估 DPO 對齊效果
- 對比 SFT vs DPO 輸出質量
- 測試不同場景的表現
- 分析對齊成功與失敗案例

#### 評估方法

1. **自動化評估**
   ```python
   # Win Rate 測試
   def compute_win_rate(model_a, model_b, test_prompts):
       """計算模型 A 相對模型 B 的勝率"""
       wins = 0
       for prompt in test_prompts:
           response_a = model_a.generate(prompt)
           response_b = model_b.generate(prompt)

           # 使用 GPT-4 或人工評判
           if judge(response_a, response_b, prompt) == 'A':
               wins += 1

       return wins / len(test_prompts)
   ```

2. **人類偏好測試**
   - A/B 測試
   - Elo Rating
   - 主觀質量評分

3. **安全性評估**
   - 有害內容檢測
   - 偏見分析
   - 拒絕不當請求能力

4. **對話質量評估**
   - 相關性 (Relevance)
   - 幫助性 (Helpfulness)
   - 無害性 (Harmlessness)

#### 評估指標

| 指標 | SFT 基準 | DPO 目標 | 測量方法 |
|------|---------|---------|----------|
| **Win Rate** | 50% (vs self) | >60% (vs SFT) | A/B 測試 |
| **Reward Margin** | 0 | >0.5 | 隱式獎勵 |
| **Helpfulness** | 3.5/5 | 4.2/5 | 人工評分 |
| **Harmlessness** | 3.8/5 | 4.5/5 | 安全評估 |

---

## 🔧 技術實現細節

### 偏好數據集範例

#### Anthropic HH-RLHF 格式
```json
{
  "chosen": "Human: How do I make pizza?\n\nAssistant: To make pizza, you'll need flour, yeast, water, tomato sauce, cheese, and toppings...",

  "rejected": "Human: How do I make pizza?\n\nAssistant: Just buy it from the store."
}
```

#### 數據準備流程
```python
def prepare_dpo_dataset(dataset, tokenizer):
    """準備 DPO 訓練數據"""
    processed = []

    for example in dataset:
        # 分離 prompt 和 response
        chosen_text = example['chosen']
        rejected_text = example['rejected']

        # 提取共同 prompt
        prompt = extract_common_prompt(chosen_text, rejected_text)
        chosen_response = chosen_text.replace(prompt, '').strip()
        rejected_response = rejected_text.replace(prompt, '').strip()

        # Tokenize
        prompt_ids = tokenizer.encode(prompt, add_special_tokens=True)
        chosen_ids = tokenizer.encode(chosen_response, add_special_tokens=False)
        rejected_ids = tokenizer.encode(rejected_response, add_special_tokens=False)

        processed.append({
            'prompt_ids': prompt_ids,
            'chosen_ids': chosen_ids,
            'rejected_ids': rejected_ids
        })

    return processed
```

### DPO 訓練最佳實踐

#### 1. 參考模型選擇
```python
# 方案 1: 使用 SFT 模型作為參考
ref_model = copy.deepcopy(sft_model)
ref_model.eval()
for param in ref_model.parameters():
    param.requires_grad = False

# 方案 2: 使用基礎模型 (如果沒有 SFT)
ref_model = AutoModelForCausalLM.from_pretrained("gpt2")
```

#### 2. Beta 參數選擇
- **β = 0.1**: 激進對齊, 可能過度優化
- **β = 0.2-0.3**: 平衡, 推薦起點
- **β = 0.5**: 保守, 保持接近參考模型

#### 3. 學習率調度
```python
# DPO 對學習率敏感
optimizer = torch.optim.AdamW(
    policy_model.parameters(),
    lr=5e-7,  # 比 SFT 小 10x
    betas=(0.9, 0.95)
)

# Cosine 退火
scheduler = get_cosine_schedule_with_warmup(
    optimizer,
    num_warmup_steps=100,
    num_training_steps=total_steps
)
```

#### 4. 訓練監控
```python
# 關鍵指標
metrics_to_track = {
    'loss': [],
    'reward_margin': [],  # chosen - rejected 獎勵差
    'accuracy': [],       # P(chosen > rejected)
    'kl_div': []          # KL(policy || ref)
}

# 理想訓練曲線:
# - loss 下降
# - reward_margin 上升
# - accuracy > 60%
# - kl_div 保持較小 (<10)
```

---

## 🚀 環境準備

### 前置要求

#### 硬體要求
- **GPU**: 16GB+ VRAM (推薦 24GB+)
  - 7B 模型需要 2 個模型同時載入 (policy + ref)
- **儲存空間**: 50GB+ (模型檢查點)

#### 軟體依賴
```bash
# 安裝 trl (Transformer Reinforcement Learning)
pip install trl

# 安裝 datasets
pip install datasets

# 驗證安裝
python -c "from trl import DPOTrainer; print('✅ TRL 可用')"
```

### 使用 TRL 庫的 DPO Trainer

```python
from trl import DPOTrainer, DPOConfig

# 配置
config = DPOConfig(
    beta=0.1,
    learning_rate=5e-7,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=16,
    num_train_epochs=1,
    max_length=512,
    max_prompt_length=256
)

# 訓練器
trainer = DPOTrainer(
    model=policy_model,
    ref_model=ref_model,
    args=config,
    train_dataset=dpo_dataset,
    tokenizer=tokenizer
)

# 訓練
trainer.train()
```

---

## 📈 性能預期

### DPO vs SFT 對比 (GPT-2 Small)

| 指標 | SFT 基準 | DPO | 改進 |
|------|---------|-----|------|
| **Helpfulness** | 3.2/5 | 4.1/5 | +28% |
| **Harmlessness** | 3.5/5 | 4.3/5 | +23% |
| **Win Rate** | 50% | 68% | +36% |
| **Perplexity** | 25.3 | 26.1 | -3% (可接受) |

### 訓練成本對比

| 方法 | 階段數 | GPU 時 | 相對成本 |
|------|--------|-------|---------|
| **RLHF (PPO)** | 3 | 100 | 100% |
| **DPO** | 2 | 40 | 40% |
| **僅 SFT** | 1 | 20 | 20% |

---

## 💡 最佳實踐

### 數據準備建議

1. **數據質量最重要**
   - 偏好對差異明顯
   - chosen 真正更好
   - rejected 具有代表性

2. **數據量建議**
   - 最少: 10K 偏好對
   - 推薦: 50K-100K
   - 大規模: 500K+ (SOTA 模型)

3. **數據平衡**
   - 不同領域均衡
   - 難度分布合理
   - 避免偏見

### 訓練技巧

1. **先 SFT 後 DPO**
   - SFT 建立基礎能力
   - DPO 優化偏好對齊
   - 效果最佳

2. **Beta 調優策略**
   ```python
   # 從大到小嘗試
   beta_values = [0.5, 0.3, 0.1, 0.05]

   # 觀察 reward margin 和 KL divergence
   # 選擇 margin 最大且 KL < 10 的 beta
   ```

3. **監控過度優化**
   ```python
   # 警告信號:
   # - KL divergence > 20 (過度偏離參考模型)
   # - Loss 持續下降但生成質量變差
   # - Accuracy > 95% (過擬合偏好數據)

   # 應對: 早停, 增大 beta, 減小學習率
   ```

---

## 🎓 學習檢查清單

完成本實驗室後，您應該能夠:

### 理論理解
- [ ] 解釋 DPO 相比 RLHF 的優勢
- [ ] 推導 DPO 損失函數
- [ ] 理解 Bradley-Terry 偏好模型
- [ ] 說明隱式獎勵的概念
- [ ] 理解 beta 參數的作用

### 實作技能
- [ ] 準備偏好數據集
- [ ] 實現 DPO 損失函數
- [ ] 訓練 DPO 對齊模型
- [ ] 監控訓練關鍵指標
- [ ] 評估對齊效果

### 應用能力
- [ ] 為項目選擇合適的對齊方法
- [ ] 收集或生成偏好數據
- [ ] 調優 DPO 超參數
- [ ] 診斷訓練問題
- [ ] 部署對齊後的模型

---

## 🚀 下一步學習

完成本實驗室後，建議繼續:

1. **Lab-1.8: ORPO Alignment**
   - Odds Ratio Preference Optimization
   - 單階段對齊 (無需 SFT)

2. **高級對齊技術**
   - Constitutional AI
   - RLAIF (AI Feedback)
   - Iterative DPO

3. **生產部署**
   - 對齊模型服務化
   - A/B 測試框架
   - 持續改進機制

---

**實驗室狀態**: 🔄 開發中
**最後更新**: 2025-10-08
**維護者**: LLM 教學專案團隊

**相關文件**:
- 理論: `01-Theory/1.3-Optimization_and_Alignment.md` (DPO 章節)
- 前置實驗: `Lab-1.4-Training_Optimization_Basics`
- 後續實驗: `Lab-1.8-ORPO_Alignment`

**相關論文**:
- [Direct Preference Optimization (DPO)](https://arxiv.org/abs/2305.18290) - NeurIPS 2023
- [Training language models to follow instructions with human feedback (RLHF)](https://arxiv.org/abs/2203.02155)
- [Constitutional AI: Harmlessness from AI Feedback](https://arxiv.org/abs/2212.08073)

**開源資源**:
- [HuggingFace TRL](https://github.com/huggingface/trl) - DPO Trainer 實現
- [Anthropic HH-RLHF Dataset](https://huggingface.co/datasets/Anthropic/hh-rlhf)
- [Zephyr-7B](https://huggingface.co/HuggingFaceH4/zephyr-7b-beta) - 使用 DPO 對齊的模型
