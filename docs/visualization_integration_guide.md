# 視覺化整合指南
## Visualization Integration Guide for PEFT Labs

**版本**: v1.0
**日期**: 2025-10-09
**用途**: 指導如何在 PEFT 實驗室中整合視覺化工具

---

## 📚 概述

本指南說明如何在現有的 PEFT 實驗室 notebooks 中整合 `common_utils.visualization` 工具。

---

## 🎯 整合位置

### 02-Train.ipynb - 訓練過程視覺化

**位置**: 在 `trainer.train()` 之後，Final Evaluation 之後

**添加的 Cell**:

#### Cell 1: Markdown 說明
```markdown
### Step 5: 視覺化訓練結果

使用 common_utils 提供的視覺化工具來展示訓練過程。
```

#### Cell 2: 視覺化代碼
```python
# 導入視覺化工具
import sys
sys.path.append('../../../..')  # 添加專案根目錄到路徑
from common_utils.visualization import plot_training_curves, plot_parameter_distribution
from common_utils.training_helpers import print_trainable_parameters

print("=" * 60)
print("訓練過程視覺化")
print("=" * 60)

# 1. 提取訓練歷史
log_history = trainer.state.log_history

# 提取損失值
train_losses = []
eval_losses = []
eval_perplexities = []

for entry in log_history:
    if 'loss' in entry and 'eval_loss' not in entry:
        train_losses.append(entry['loss'])
    if 'eval_loss' in entry:
        eval_losses.append(entry['eval_loss'])
    if 'eval_perplexity' in entry:
        eval_perplexities.append(entry['eval_perplexity'])

# 2. 繪製訓練曲線
history = {
    'train_loss': train_losses,
    'eval_loss': eval_losses,
    'eval_perplexity': eval_perplexities
}

plot_training_curves(
    history,
    title="LoRA 訓練過程",
    save_path="./lora-llama2-7b-guanaco/training_curves.png"
)

print("✅ 訓練曲線已生成")

# 3. 參數分佈視覺化
model_stats = print_trainable_parameters(peft_model, verbose=False)

plot_parameter_distribution(
    model_stats,
    save_path="./lora-llama2-7b-guanaco/parameter_distribution.png"
)

print("✅ 參數分佈圖已生成")
print("\n所有視覺化圖表已保存至: ./lora-llama2-7b-guanaco/")
```

---

### 03-Inference.ipynb - 推理性能視覺化

**位置**: 在推理測試之後

#### Cell: 推理性能對比
```python
from common_utils.visualization import plot_inference_benchmark

# 測試推理性能
import time

def benchmark_inference(model, tokenizer, prompt, num_runs=5):
    """簡單的推理性能測試"""
    latencies = []

    for _ in range(num_runs):
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

        start = time.time()
        with torch.no_grad():
            outputs = model.generate(**inputs, max_new_tokens=50)
        latency = (time.time() - start) * 1000  # ms

        latencies.append(latency)

    return {
        'latency_ms': sum(latencies) / len(latencies),
        'throughput': 50 / (sum(latencies) / len(latencies) / 1000)  # tokens/s
    }

# 測試
test_prompt = "What is machine learning?"

base_result = benchmark_inference(base_model, tokenizer, test_prompt)
lora_result = benchmark_inference(peft_model, tokenizer, test_prompt)

# 視覺化對比
benchmark_results = {
    'Base Model': base_result,
    'LoRA Fine-tuned': lora_result
}

plot_inference_benchmark(
    benchmark_results,
    save_path="./lora-llama2-7b-guanaco/inference_benchmark.png"
)

print("✅ 推理性能對比圖已生成")
```

---

### 04-Merge_and_Deploy.ipynb - 部署前檢查

**位置**: 合併完成後

#### Cell: 模型大小對比
```python
import os
from common_utils.visualization import plt

# 統計模型大小
def get_model_size(path):
    """計算目錄中所有文件的總大小"""
    total_size = 0
    for dirpath, dirnames, filenames in os.walk(path):
        for filename in filenames:
            filepath = os.path.join(dirpath, filename)
            total_size += os.path.getsize(filepath)
    return total_size / (1024**3)  # GB

adapter_size = get_model_size("./lora-llama2-7b-guanaco")
merged_size = get_model_size("./lora-llama2-7b-guanaco-merged")

# 視覺化
models = ['LoRA Adapter', 'Merged Model', 'Original Model (參考)']
sizes = [adapter_size, merged_size, 13.5]  # Llama-2-7B FP16 約 13.5GB
colors = ['#2ecc71', '#3498db', '#95a5a6']

fig, ax = plt.subplots(figsize=(10, 6))
bars = ax.bar(models, sizes, color=colors, alpha=0.8)

ax.set_ylabel('模型大小 (GB)', fontsize=12, fontweight='bold')
ax.set_title('LoRA 模型大小對比', fontsize=14, fontweight='bold')
ax.grid(axis='y', alpha=0.3)

# 添加數值標籤
for bar, size in zip(bars, sizes):
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height,
           f'{size:.2f}GB',
           ha='center', va='bottom', fontsize=11, fontweight='bold')

# 添加效率說明
ax.text(0.5, -0.15,
       f'LoRA Adapter 僅佔原模型大小的 {adapter_size/13.5*100:.2f}%',
       transform=ax.transAxes, ha='center', fontsize=10, style='italic')

plt.tight_layout()
plt.savefig("./lora-llama2-7b-guanaco-merged/model_size_comparison.png", dpi=300)
plt.show()

print(f"✅ LoRA Adapter: {adapter_size:.2f}GB")
print(f"✅ 合併模型: {merged_size:.2f}GB")
print(f"✅ 效率: Adapter 僅佔 {adapter_size/merged_size*100:.1f}%")
```

---

## 🔧 標準化模板

### 模板 1: 訓練視覺化 (所有 02-Train.ipynb)

```python
# ========== 在 trainer.train() 之後添加 ==========

# 導入視覺化工具
from common_utils.visualization import plot_training_curves
from common_utils.training_helpers import analyze_training_results

# 視覺化訓練過程
log_history = trainer.state.log_history

# 提取指標
train_loss = [e['loss'] for e in log_history if 'loss' in e and 'eval' not in str(e)]
eval_loss = [e['eval_loss'] for e in log_history if 'eval_loss' in e]

history = {'train_loss': train_loss, 'eval_loss': eval_loss}

# 添加其他評估指標
for entry in log_history:
    for key in entry:
        if key.startswith('eval_') and key not in history:
            if key not in history:
                history[key] = []
            history[key].append(entry[key])

# 繪製
plot_training_curves(
    history,
    title=f"{METHOD_NAME} 訓練過程",  # 替換 METHOD_NAME
    save_path=f"./{OUTPUT_DIR}/training_curves.png"
)

# 分析結果
results = analyze_training_results(trainer, f"./{OUTPUT_DIR}")
```

### 模板 2: 參數分佈 (所有 02-Train.ipynb 或 01-Setup.ipynb)

```python
from common_utils.visualization import plot_parameter_distribution
from common_utils.training_helpers import print_trainable_parameters

# 獲取參數統計
model_stats = print_trainable_parameters(model, verbose=True)

# 視覺化
plot_parameter_distribution(
    model_stats,
    save_path=f"./{OUTPUT_DIR}/parameter_distribution.png"
)
```

### 模板 3: 方法對比 (可選，在對比實驗中)

```python
from common_utils.visualization import plot_peft_comparison

results = [
    {'method': 'LoRA r=8', 'trainable_params_%': 0.25, 'performance': 85.2, 'training_time': 120},
    {'method': 'LoRA r=16', 'trainable_params_%': 0.48, 'performance': 86.1, 'training_time': 135},
    {'method': 'LoRA r=32', 'trainable_params_%': 0.95, 'performance': 86.5, 'training_time': 155},
]

plot_peft_comparison(
    results,
    metrics=['trainable_params_%', 'performance', 'training_time']
)
```

---

## 📋 整合檢查清單

### Lab-01 (LoRA) 整合
- [ ] 02-Train.ipynb: 添加訓練曲線視覺化
- [ ] 02-Train.ipynb: 添加參數分佈圖
- [ ] 03-Inference.ipynb: 添加推理性能對比
- [ ] 04-Merge_and_Deploy.ipynb: 添加模型大小對比

### Lab-02 (AdapterLayers) 整合
- [ ] 02-Train.ipynb: 訓練曲線
- [ ] 03-Inference.ipynb: 多任務性能對比
- [ ] 04-Merge_and_Deploy.ipynb: Adapter 結構視覺化

### Lab-05 (IA3) 整合
- [ ] 02-Train.ipynb: 訓練曲線
- [ ] 02-Train.ipynb: IA3 vs LoRA 參數對比
- [ ] 03-Inference.ipynb: 推理性能

### 其他 Labs (Lab-03, 04, 06, 07, 08)
- [ ] 按相同模式整合

---

## 🚀 實施計劃

### Phase 1: 試點 (本週)
1. Lab-01 (LoRA) - 完整整合
2. Lab-05 (IA3) - 完整整合
3. 驗證工具可用性

**預估工時**: 2-3 小時

### Phase 2: 推廣 (下週)
1. Lab-02, Lab-08 整合
2. 優化模板
3. 形成標準流程

**預估工時**: 2-3 小時

### Phase 3: 完成 (第三週)
1. 剩餘實驗室整合
2. 品質檢查
3. 更新文檔

**預估工時**: 2-3 小時

**總工時**: 6-9 小時

---

## 💡 注意事項

### 路徑問題
notebooks 在不同層級，需要正確添加專案根目錄到 Python 路徑:

```python
import sys
import os

# 方法 1: 相對路徑
sys.path.append(os.path.abspath('../../../..'))

# 方法 2: 動態計算
current_dir = os.path.dirname(os.path.abspath('__file__'))
project_root = os.path.abspath(os.path.join(current_dir, '../../../..'))
sys.path.append(project_root)

# 然後導入
from common_utils import plot_training_curves
```

### 依賴檢查
確保所有 notebooks 環境包含:
```bash
pip install matplotlib seaborn pandas
```

### 錯誤處理
如果視覺化失敗，不應中斷訓練流程:

```python
try:
    from common_utils.visualization import plot_training_curves
    plot_training_curves(history)
except Exception as e:
    print(f"⚠️  視覺化失敗: {e}")
    print("訓練結果已保存，請手動查看 log_history")
```

---

## 📊 預期效果

### 整合前
- 學生只能看到文字輸出的 loss 值
- 無法直觀理解訓練趨勢
- 難以比較不同配置

### 整合後
- ✅ 清晰的訓練/驗證損失曲線
- ✅ 困惑度變化趨勢圖
- ✅ 參數效率視覺化
- ✅ 推理性能對比圖表
- ✅ 一致的視覺風格

---

## 🎓 教學價值

**學習體驗提升**:
- 視覺化學習效果更佳
- 快速識別訓練問題
- 直觀理解參數效率概念

**實用性提升**:
- 可用於實際項目
- 標準化的性能報告
- 易於分享與展示

---

**文檔版本**: v1.0
**維護者**: LLM 教學專案團隊
**下一步**: 在 Lab-01 進行試點整合
