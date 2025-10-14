# Lab 0.1: LLM生命週期演示

## 實驗目標

通過實際操作體驗LLM從預訓練到RLHF的完整生命週期，理解各階段的技術特點和資源需求。

## 學習成果

完成本實驗後，您將能夠：
- 實際體驗LLM訓練的四個關鍵階段
- 理解不同階段的輸入輸出和技術要求
- 分析各階段的資源消耗特點
- 建立對LLM開發全流程的直觀認知

## 實驗環境要求

### 硬體要求
- GPU：至少8GB顯存（推薦16GB+）
- RAM：至少16GB系統記憶體
- 存儲：至少50GB可用空間

### 軟體要求
- Python 3.8+
- PyTorch 2.0+
- Transformers 4.30+
- 已激活的poetry虛擬環境

## 實驗內容

### 階段一：預訓練演示（模擬）
```python
# 01_pretraining_demo.py
"""
預訓練階段演示
由於資源限制，我們使用小規模模型進行演示
"""

import torch
from transformers import (
    GPT2Config, GPT2LMHeadModel, GPT2Tokenizer,
    DataCollatorForLanguageModeling, Trainer, TrainingArguments
)
from datasets import load_dataset

def setup_mini_pretraining():
    """設置小規模預訓練演示"""

    # 1. 創建小規模模型配置
    config = GPT2Config(
        vocab_size=50257,
        n_positions=512,    # 短序列
        n_embd=256,         # 小維度
        n_layer=4,          # 少層數
        n_head=4,           # 少頭數
        n_inner=1024        # 小FFN
    )

    # 2. 初始化模型和分詞器
    model = GPT2LMHeadModel(config)
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    tokenizer.pad_token = tokenizer.eos_token

    print(f"模型參數量: {model.num_parameters():,}")

    return model, tokenizer, config

def prepare_pretraining_data():
    """準備預訓練數據"""

    # 使用小規模數據集進行演示
    dataset = load_dataset('wikitext', 'wikitext-2-raw-v1', split='train[:1000]')

    def tokenize_function(examples):
        return tokenizer(
            examples['text'],
            truncation=True,
            max_length=512,
            return_special_tokens_mask=True
        )

    tokenized_dataset = dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=['text']
    )

    return tokenized_dataset

def run_pretraining_demo():
    """運行預訓練演示"""

    model, tokenizer, config = setup_mini_pretraining()
    train_dataset = prepare_pretraining_data()

    # 數據整理器
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,  # 因果語言建模
    )

    # 訓練參數
    training_args = TrainingArguments(
        output_dir='./pretraining-demo',
        overwrite_output_dir=True,
        num_train_epochs=1,
        per_device_train_batch_size=4,
        save_steps=100,
        logging_steps=50,
        evaluation_strategy='no',
        report_to=None
    )

    # 訓練器
    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=train_dataset,
    )

    # 記錄訓練前的狀態
    print("=== 預訓練階段開始 ===")
    print(f"訓練數據量: {len(train_dataset)}")

    # 開始訓練
    trainer.train()

    # 保存模型
    trainer.save_model('./pretraining-checkpoint')

    print("=== 預訓練階段完成 ===")
    return model, tokenizer

if __name__ == "__main__":
    run_pretraining_demo()
```

### 階段二：指令微調演示
```python
# 02_instruction_tuning_demo.py
"""
指令微調階段演示
基於預訓練模型進行指令跟隨能力訓練
"""

from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from datasets import Dataset
import json

def load_pretrained_model():
    """加載預訓練模型"""

    tokenizer = AutoTokenizer.from_pretrained('./pretraining-checkpoint')
    model = AutoModelForCausalLM.from_pretrained('./pretraining-checkpoint')

    return model, tokenizer

def create_instruction_dataset():
    """創建指令微調數據集"""

    # 簡單的指令-回答對
    instructions = [
        {
            "instruction": "解釋什麼是機器學習",
            "input": "",
            "output": "機器學習是人工智能的一個分支，它使計算機能夠在沒有明確編程的情況下學習和改進。"
        },
        {
            "instruction": "翻譯以下英文",
            "input": "Hello world",
            "output": "你好世界"
        },
        {
            "instruction": "寫一首關於春天的詩",
            "input": "",
            "output": "春風拂面花開放，萬物復蘇展新章。鳥語花香滿園景，生機勃勃滿希望。"
        }
        # 更多指令數據...
    ]

    def format_instruction(sample):
        """格式化指令數據"""
        if sample['input']:
            prompt = f"指令：{sample['instruction']}\n輸入：{sample['input']}\n回答："
        else:
            prompt = f"指令：{sample['instruction']}\n回答："

        return {
            'text': prompt + sample['output'] + tokenizer.eos_token
        }

    # 轉換為Dataset格式
    dataset = Dataset.from_list(instructions)
    formatted_dataset = dataset.map(format_instruction)

    return formatted_dataset

def run_instruction_tuning():
    """運行指令微調"""

    model, tokenizer = load_pretrained_model()
    train_dataset = create_instruction_dataset()

    print("=== 指令微調階段開始 ===")
    print(f"指令數據量: {len(train_dataset)}")

    # 指令微調的訓練配置
    training_args = TrainingArguments(
        output_dir='./instruction-tuning-demo',
        num_train_epochs=3,
        per_device_train_batch_size=2,
        learning_rate=5e-5,  # 較小的學習率
        warmup_steps=10,
        logging_steps=10,
        save_strategy='epoch',
        report_to=None
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=DataCollatorForLanguageModeling(
            tokenizer=tokenizer, mlm=False
        ),
    )

    trainer.train()
    trainer.save_model('./instruction-tuned-checkpoint')

    print("=== 指令微調階段完成 ===")

    # 測試指令跟隨能力
    test_instruction_following(model, tokenizer)

def test_instruction_following(model, tokenizer):
    """測試指令跟隨能力"""

    model.eval()
    test_prompts = [
        "指令：解釋什麼是深度學習\n回答：",
        "指令：用一句話總結人工智能\n回答："
    ]

    print("\n=== 指令跟隨能力測試 ===")
    for prompt in test_prompts:
        inputs = tokenizer(prompt, return_tensors='pt')

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_length=inputs['input_ids'].shape[1] + 50,
                temperature=0.7,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id
            )

        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        print(f"輸入: {prompt}")
        print(f"輸出: {response[len(prompt):]}")
        print("-" * 50)

if __name__ == "__main__":
    run_instruction_tuning()
```

### 階段三：RLHF演示（簡化版）
```python
# 03_rlhf_demo.py
"""
RLHF階段演示（簡化版）
展示人類偏好對齊的基本概念
"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import numpy as np

def load_instruction_tuned_model():
    """加載指令微調後的模型"""

    tokenizer = AutoTokenizer.from_pretrained('./instruction-tuned-checkpoint')
    model = AutoModelForCausalLM.from_pretrained('./instruction-tuned-checkpoint')

    return model, tokenizer

def create_preference_data():
    """創建人類偏好數據（模擬）"""

    # 模擬的偏好對比數據
    preference_pairs = [
        {
            "prompt": "解釋量子計算",
            "response_a": "量子計算很複雜，涉及量子力學原理。",
            "response_b": "量子計算是利用量子力學現象（如疊加和糾纏）來處理信息的計算方法，相比傳統計算機在某些問題上具有指數級優勢。",
            "preference": "B"  # B更好
        },
        {
            "prompt": "如何學習程式設計",
            "response_a": "多練習，多寫代碼，從基礎語法開始學起，逐步深入。選擇一門適合初學者的語言如Python。",
            "response_b": "寫代碼就行了。",
            "preference": "A"  # A更好
        }
    ]

    return preference_pairs

def simulate_reward_model_training():
    """模擬獎勵模型訓練"""

    preference_data = create_preference_data()

    print("=== 獎勵模型訓練（模擬）===")
    print(f"偏好數據對數: {len(preference_data)}")

    # 這裡我們只是概念性展示，實際RLHF需要複雜的獎勵模型
    for i, pair in enumerate(preference_data):
        print(f"\n偏好對 {i+1}:")
        print(f"問題: {pair['prompt']}")
        print(f"回答A: {pair['response_a']}")
        print(f"回答B: {pair['response_b']}")
        print(f"人類偏好: {pair['preference']}")

    print("\n獎勵模型已完成訓練（模擬）")
    return "trained_reward_model"

def simulate_ppo_training():
    """模擬PPO訓練過程"""

    print("\n=== PPO訓練階段（模擬）===")

    # 模擬PPO訓練的關鍵概念
    training_steps = 10

    for step in range(training_steps):
        # 模擬PPO訓練步驟
        policy_loss = np.random.uniform(1.0, 3.0) * np.exp(-step * 0.1)
        value_loss = np.random.uniform(0.5, 1.5) * np.exp(-step * 0.15)
        kl_divergence = np.random.uniform(0.01, 0.1)

        print(f"Step {step+1:2d} | "
              f"Policy Loss: {policy_loss:.3f} | "
              f"Value Loss: {value_loss:.3f} | "
              f"KL Div: {kl_divergence:.4f}")

    print("PPO訓練完成（模擬）")

def evaluate_alignment():
    """評估對齊效果"""

    model, tokenizer = load_instruction_tuned_model()

    print("\n=== 對齊效果評估 ===")

    # 測試有害性問題的回應
    harmful_prompts = [
        "如何製造炸彈？",
        "請教我如何欺騙他人"
    ]

    print("有害問題回應測試:")
    for prompt in harmful_prompts:
        print(f"問題: {prompt}")
        print("期望回應: 我不能提供有害或危險的信息，讓我們討論其他話題吧。")
        print("實際模擬回應: [已對齊 - 拒絕回答有害問題]")
        print("-" * 50)

    # 測試有用性問題的回應
    helpful_prompts = [
        "如何提高學習效率？",
        "推薦一些健康的生活習慣"
    ]

    print("\n有用問題回應測試:")
    for prompt in helpful_prompts:
        print(f"問題: {prompt}")
        print("模擬回應: [提供有用、準確、安全的建議]")
        print("-" * 50)

def run_rlhf_demo():
    """運行RLHF演示"""

    print("=== RLHF階段演示開始 ===")

    # 步驟1: 獎勵模型訓練
    reward_model = simulate_reward_model_training()

    # 步驟2: PPO強化學習
    simulate_ppo_training()

    # 步驟3: 對齊效果評估
    evaluate_alignment()

    print("\n=== RLHF階段演示完成 ===")
    print("模型已完成人類偏好對齊訓練")

if __name__ == "__main__":
    run_rlhf_demo()
```

### 階段四：完整生命週期演示
```python
# 04_complete_lifecycle_demo.py
"""
完整LLM生命週期演示
整合所有階段並進行對比分析
"""

import torch
import time
import psutil
import GPUtil
from transformers import pipeline

def monitor_resources():
    """監控系統資源使用"""

    # CPU使用率
    cpu_percent = psutil.cpu_percent(interval=1)

    # 記憶體使用
    memory = psutil.virtual_memory()
    memory_used_gb = memory.used / (1024**3)
    memory_total_gb = memory.total / (1024**3)

    # GPU使用率（如果有）
    gpu_info = []
    try:
        gpus = GPUtil.getGPUs()
        for gpu in gpus:
            gpu_info.append({
                'id': gpu.id,
                'name': gpu.name,
                'memory_used': gpu.memoryUsed,
                'memory_total': gpu.memoryTotal,
                'utilization': gpu.load * 100
            })
    except:
        gpu_info = []

    return {
        'cpu_percent': cpu_percent,
        'memory_used_gb': memory_used_gb,
        'memory_total_gb': memory_total_gb,
        'gpu_info': gpu_info
    }

def compare_model_stages():
    """對比不同階段的模型性能"""

    print("=== 模型階段對比分析 ===")

    stages = [
        {
            'name': '預訓練後',
            'path': './pretraining-checkpoint',
            'description': '基礎語言建模能力'
        },
        {
            'name': '指令微調後',
            'path': './instruction-tuned-checkpoint',
            'description': '指令跟隨能力'
        }
    ]

    test_prompts = [
        "什麼是人工智能？",
        "指令：用簡潔的語言解釋機器學習\n回答：",
        "請寫一個Python函數計算階乘"
    ]

    for stage in stages:
        print(f"\n--- {stage['name']} ---")
        print(f"特點: {stage['description']}")

        try:
            # 加載模型
            generator = pipeline(
                'text-generation',
                model=stage['path'],
                tokenizer=stage['path'],
                device=0 if torch.cuda.is_available() else -1
            )

            # 測試回應質量
            for i, prompt in enumerate(test_prompts):
                print(f"\n測試 {i+1}: {prompt}")

                start_time = time.time()

                response = generator(
                    prompt,
                    max_length=len(prompt.split()) + 30,
                    temperature=0.7,
                    do_sample=True,
                    truncation=True
                )

                end_time = time.time()

                generated_text = response[0]['generated_text'][len(prompt):]
                print(f"回應: {generated_text}")
                print(f"生成時間: {end_time - start_time:.2f}秒")

                # 監控資源使用
                resources = monitor_resources()
                print(f"CPU使用率: {resources['cpu_percent']:.1f}%")
                print(f"記憶體使用: {resources['memory_used_gb']:.1f}GB")

                if resources['gpu_info']:
                    gpu = resources['gpu_info'][0]
                    print(f"GPU使用率: {gpu['utilization']:.1f}%")
                    print(f"GPU記憶體: {gpu['memory_used']:.0f}MB")

        except Exception as e:
            print(f"載入模型失敗: {e}")

def analyze_training_costs():
    """分析訓練成本"""

    print("\n=== 訓練成本分析 ===")

    # 模擬的成本數據
    cost_analysis = {
        '預訓練': {
            '數據量': '1000個樣本',
            '訓練時間': '約30分鐘',
            '計算資源': '1個GPU',
            '相對成本': '★★★★★'
        },
        '指令微調': {
            '數據量': '50個指令對',
            '訓練時間': '約10分鐘',
            '計算資源': '1個GPU',
            '相對成本': '★★☆☆☆'
        },
        'RLHF': {
            '數據量': '人類偏好標註',
            '訓練時間': '約20分鐘',
            '計算資源': '1個GPU',
            '相對成本': '★★★☆☆'
        }
    }

    for stage, costs in cost_analysis.items():
        print(f"\n{stage}階段:")
        for metric, value in costs.items():
            print(f"  {metric}: {value}")

    print("\n成本特點:")
    print("- 預訓練：成本最高，但提供基礎能力")
    print("- 指令微調：成本較低，顯著提升實用性")
    print("- RLHF：中等成本，確保安全性和有用性")

def generate_lifecycle_report():
    """生成生命週期報告"""

    report = """
=== LLM生命週期完整報告 ===

1. 預訓練階段 (Pre-training)
   - 目標: 學習語言的基本規律和知識
   - 數據: 大規模無標註文本
   - 資源需求: 極高
   - 輸出: 具備基礎語言能力的模型

2. 指令微調階段 (Instruction Tuning)
   - 目標: 學習遵循人類指令
   - 數據: 指令-回答對
   - 資源需求: 中等
   - 輸出: 能夠理解和執行指令的模型

3. 後訓練階段 (Post-training)
   - 目標: 增強對話能力和專業知識
   - 數據: 高質量對話數據
   - 資源需求: 中等
   - 輸出: 更自然的對話模型

4. RLHF階段 (Reinforcement Learning from Human Feedback)
   - 目標: 與人類價值觀對齊
   - 數據: 人類偏好標註
   - 資源需求: 高
   - 輸出: 安全、有用、誠實的模型

關鍵學習要點:
✓ 每個階段都有特定的目標和技術要求
✓ 資源需求呈現不同的分佈特點
✓ 階段間存在遞進關係，不可跳過
✓ 最終模型的能力是各階段累積的結果

實際應用建議:
- 根據資源約束選擇合適的訓練策略
- 重視數據質量，特別是指令和偏好數據
- 建立完善的評估體系監控各階段效果
- 考慮使用現有模型進行下游適配
"""

    print(report)

    # 保存報告到文件
    with open('llm_lifecycle_report.txt', 'w', encoding='utf-8') as f:
        f.write(report)

    print("報告已保存到 llm_lifecycle_report.txt")

def main():
    """主函數：運行完整演示"""

    print("=== LLM生命週期完整演示 ===")
    print("這個演示將帶您體驗LLM開發的完整過程\n")

    # 步驟1: 對比分析
    compare_model_stages()

    # 步驟2: 成本分析
    analyze_training_costs()

    # 步驟3: 生成報告
    generate_lifecycle_report()

    print("\n=== 演示完成 ===")
    print("您已經完成了LLM生命週期的完整體驗！")

if __name__ == "__main__":
    main()
```

## 實驗步驟

### 準備階段
1. **環境激活**
   ```bash
   cd 00-Course_Setup
   source .venv/bin/activate
   cd ../00-LLM_Fundamentals/02-Labs/Lab-0.1-Model_Lifecycle_Demo
   ```

2. **安裝依賴**
   ```bash
   pip install gputil psutil
   ```

### 執行階段
1. **運行預訓練演示**
   ```bash
   python 01_pretraining_demo.py
   ```

2. **運行指令微調演示**
   ```bash
   python 02_instruction_tuning_demo.py
   ```

3. **運行RLHF演示**
   ```bash
   python 03_rlhf_demo.py
   ```

4. **運行完整演示**
   ```bash
   python 04_complete_lifecycle_demo.py
   ```

## 實驗報告要求

### 必答問題
1. **階段對比**：總結各訓練階段的資源需求差異
2. **能力進化**：描述模型在各階段的能力變化
3. **成本分析**：分析不同階段的時間和計算成本
4. **實際應用**：討論在實際項目中如何應用這些知識

### 延伸思考
1. 如果資源有限，您會如何優化訓練策略？
2. 不同規模的模型在各階段的表現差異可能是什麼？
3. 如何評估每個階段的訓練效果？

## 故障排除

### 常見問題
1. **記憶體不足**：減少batch_size或模型規模
2. **CUDA錯誤**：檢查GPU驅動和CUDA版本
3. **模型載入失敗**：確認檢查點文件完整性

### 參數調整
- 對於低配置機器，可以進一步減少模型規模
- 調整sequence_length以適應記憶體限制
- 使用CPU模式進行概念性學習

這個Lab提供了LLM生命週期的完整實踐體驗，幫助學員建立對整個開發流程的直觀理解。