# Lab 3.1: Post-Training Quantization - 訓練後量化實戰

## 概述

**訓練後量化 (Post-Training Quantization, PTQ)** 是最實用的模型壓縮技術之一，能夠在不重新訓練的情況下將預訓練模型轉換為低精度版本。本實驗將深入探討 PTQ 的原理，並實際操作使用不同的量化策略來壓縮大型語言模型。

PTQ 的核心優勢在於：**無需重新訓練、實施簡單、壓縮效果顯著**，特別適用於已有預訓練模型的快速部署場景。

![量化效果示意圖](https://docs.nvidia.com/deeplearning/tensorrt/developer-guide/graphics/int8-calibration.png)

---

## 1. 技術背景與動機

### 1.1 量化的必要性

現代大型語言模型面臨的部署挑戰：
- **記憶體瓶頸**：Llama-2-7B 以 FP32 格式需要 28GB 記憶體
- **推理成本**：高精度浮點運算消耗大量計算資源
- **部署限制**：邊緣設備無法承載大型模型
- **服務成本**：雲端部署的經濟效益問題

### 1.2 PTQ vs QAT 比較

| 方面 | Post-Training Quantization | Quantization-Aware Training |
|:---|:---|:---|
| **實施難度** | 低（無需重新訓練） | 高（需要完整訓練流程） |
| **精度保持** | 90-98% | 95-99% |
| **時間成本** | 分鐘級 | 小時到天級 |
| **適用場景** | 快速部署、預訓練模型 | 精度敏感應用 |

---

## 2. PTQ 核心原理

### 2.1 量化映射數學基礎

**線性量化公式**：
```
q = round((r - zero_point) / scale)
r_hat = scale * (q - zero_point)
```

其中：
- `r`: 原始浮點值
- `q`: 量化後整數值
- `scale`: 縮放因子
- `zero_point`: 零點偏移

### 2.2 量化參數計算

**對稱量化** (推薦用於權重)：
```python
def symmetric_quantization(tensor, bits=8):
    max_val = torch.max(torch.abs(tensor))
    scale = max_val / (2**(bits-1) - 1)
    quantized = torch.clamp(torch.round(tensor / scale),
                           -(2**(bits-1)), 2**(bits-1)-1)
    return quantized, scale
```

**非對稱量化** (推薦用於激活)：
```python
def asymmetric_quantization(tensor, bits=8):
    min_val, max_val = torch.min(tensor), torch.max(tensor)
    scale = (max_val - min_val) / (2**bits - 1)
    zero_point = torch.round(-min_val / scale)
    quantized = torch.clamp(torch.round(tensor / scale + zero_point),
                           0, 2**bits - 1)
    return quantized, scale, zero_point
```

### 2.3 校準策略

**1. 最小-最大校準**：
- 使用激活值的最小最大值確定量化範圍
- 實現簡單，但對異常值敏感

**2. 百分位校準**：
- 使用 99.9% 百分位值避免異常值影響
- 更穩定，但可能損失部分信息

**3. KL散度校準**：
- 最小化原始分佈與量化分佈的 KL 散度
- 精度最高，但計算複雜

---

## 3. 實驗內容設計

### 3.1 實驗環境設置
- **硬體要求**：NVIDIA GPU (8GB+ VRAM)
- **軟體框架**：PyTorch, ONNX Runtime, Intel Neural Compressor
- **目標模型**：BERT-Base, GPT-2, Llama-2-7B (量化版)

### 3.2 實驗流程設計

#### Notebook 01: Setup and Basic Quantization
**學習目標**：
- 理解量化的基本概念和數學原理
- 實現基礎的線性量化算法
- 對比 FP32 vs INT8 的精度差異

**核心內容**：
```python
# 基礎量化實現
class BasicQuantizer:
    def __init__(self, bits=8):
        self.bits = bits

    def calibrate(self, data):
        # 計算量化參數
        pass

    def quantize(self, tensor):
        # 執行量化
        pass

    def dequantize(self, q_tensor):
        # 執行反量化
        pass
```

#### Notebook 02: Static Quantization
**學習目標**：
- 掌握靜態量化的完整流程
- 學習使用校準數據集計算量化參數
- 實現端到端的模型量化

**核心內容**：
- 校準數據集準備
- 量化參數計算與保存
- 量化模型推理測試

#### Notebook 03: Dynamic Quantization
**學習目標**：
- 理解動態量化的適用場景
- 實現權重量化 + 激活動態量化
- 對比靜態 vs 動態量化的效果

**核心內容**：
- 動態量化實現
- 推理速度與精度權衡
- 記憶體使用分析

#### Notebook 04: Advanced PTQ Techniques
**學習目標**：
- 學習高級校準技術 (KL divergence, Percentile)
- 實現混合精度量化策略
- 掌握量化模型的部署優化

**核心內容**：
- 高級校準算法實現
- 敏感層分析與混合精度
- ONNX Runtime 部署

---

## 4. 實驗評估指標

### 4.1 壓縮效果評估
```python
# 模型大小對比
original_size = get_model_size(original_model)
quantized_size = get_model_size(quantized_model)
compression_ratio = original_size / quantized_size

# 推理速度測試
latency_original = benchmark_inference(original_model, test_data)
latency_quantized = benchmark_inference(quantized_model, test_data)
speedup = latency_original / latency_quantized
```

### 4.2 精度保持評估
```python
# 任務特定指標
accuracy_drop = accuracy_original - accuracy_quantized
perplexity_increase = perplexity_quantized / perplexity_original

# 輸出分佈相似度
kl_divergence = compute_kl_div(original_outputs, quantized_outputs)
cosine_similarity = compute_cosine_sim(original_features, quantized_features)
```

---

## 5. 預期學習成果

### 5.1 技術能力
- **理論理解**：掌握量化的數學原理和實現細節
- **實踐技能**：能夠獨立實施各種 PTQ 策略
- **工程能力**：具備量化模型的部署和優化經驗
- **評估能力**：建立全面的量化效果評估體系

### 5.2 性能指標預期
| 模型 | 量化精度 | 壓縮比 | 推理加速 | 精度保持 |
|:---|:---|:---|:---|:---|
| **BERT-Base** | INT8 | 4x | 2.5x | >98% |
| **GPT-2** | INT8 | 4x | 2.2x | >96% |
| **Llama-2-7B** | INT8 | 4x | 1.8x | >95% |

### 5.3 實際應用場景
- **雲端部署**：降低推理服務的硬體成本
- **邊緣計算**：在資源受限設備上部署 LLM
- **移動應用**：智慧手機上的本地 AI 助手
- **實時系統**：低延遲的對話和推薦系統

---

## 6. 故障排除與最佳實踐

### 6.1 常見問題
**Q: 量化後精度大幅下降怎麼辦？**
A: 檢查校準數據的代表性，嘗試使用更多校準樣本或高級校準方法

**Q: 量化模型推理速度沒有提升？**
A: 確認硬體支持 INT8 運算，檢查推理框架的量化優化設置

**Q: 某些層量化效果很差？**
A: 對敏感層使用混合精度，保持關鍵層為 FP16 或 FP32

### 6.2 最佳實踐
1. **校準數據選擇**：使用與實際應用分佈相近的數據
2. **逐步量化**：先量化不敏感的層，再處理敏感層
3. **性能監控**：建立完整的精度和速度監控體系
4. **硬體適配**：針對目標硬體選擇最優量化策略

---

## 7. 延伸學習

### 7.1 進階技術
- **QAT (Quantization-Aware Training)**：更高精度的量化方法
- **混合精度**：自動化的精度選擇策略
- **硬體特定優化**：針對 TPU、ASIC 的量化策略

### 7.2 相關資源
- **論文閱讀**：Jacob et al. "Quantization and Training of Neural Networks"
- **開源工具**：Intel Neural Compressor, ONNX Quantization
- **硬體文檔**：NVIDIA TensorRT, Intel OpenVINO

---

**實驗設計者**: Model Compression Lab Team
**技術等級**: ⭐⭐⭐ (中級)
**預計完成時間**: 6-8 小時
**前置需求**: PyTorch 基礎、深度學習概念
**推薦後續**: Lab-3.2 (Quantization-Aware Training)