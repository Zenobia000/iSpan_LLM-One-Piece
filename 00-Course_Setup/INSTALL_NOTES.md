# 安裝注意事項
## Installation Notes

**最後更新**: 2025-10-09

---

## ⚠️ Poetry 依賴解析問題

### 問題描述
Poetry 2.1.4 在解析某些依賴時出現 `Could not parse version constraint: <empty>` 錯誤。

### 解決方案

**方法 1: 使用 pip 安裝可選依賴** (推薦)

```bash
# 1. 安裝核心依賴 (Poetry)
cd 00-Course_Setup
poetry install

# 2. 激活虛擬環境
source .venv/bin/activate

# 3. 使用 pip 安裝可選依賴
pip install vllm>=0.6.0
pip install flash-attn --no-build-isolation
```

**方法 2: 直接使用 pip** (替代方案)

```bash
# 1. 創建虛擬環境
python3.10 -m venv .venv
source .venv/bin/activate

# 2. 安裝 PyTorch (CUDA 12.1)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# 3. 安裝核心依賴
pip install transformers>=4.57 peft>=0.7 datasets>=2.14 accelerate>=0.24
pip install bitsandbytes sentencepiece protobuf

# 4. 安裝開發工具
pip install jupyterlab ipywidgets matplotlib seaborn tqdm pandas scikit-learn

# 5. 安裝可選依賴
pip install vllm>=0.6.0
pip install fastapi uvicorn
pip install flash-attn --no-build-isolation  # 需要編譯
```

---

## 📦 已安裝的依賴 (當前環境)

### 核心框架
- PyTorch 2.5.1+cu121
- Transformers 4.57+
- PEFT 0.7+
- Datasets 2.14+

### 開發工具
- JupyterLab 4.4+
- Matplotlib, Seaborn
- Pandas, NumPy

### 可選依賴
- vLLM 0.7.3 (已安裝)
- flash-attn (需手動安裝)

---

## 🔧 Flash-Attention 安裝

**要求**:
- CUDA 11.6+
- GPU Compute Capability ≥ 7.5 (Turing 架構以上)
- 足夠的編譯資源 (RAM >16GB 建議)

**安裝命令**:
```bash
pip install flash-attn --no-build-isolation
```

**如果編譯失敗**:
```bash
# 限制並行編譯
MAX_JOBS=2 pip install flash-attn --no-build-isolation

# 或從 wheel 安裝 (如果可用)
pip install flash-attn --find-links https://github.com/Dao-AILab/flash-attention/releases
```

---

## ✅ 驗證安裝

```python
# 測試核心依賴
import torch
import transformers
import peft
print(f"PyTorch: {torch.__version__}")
print(f"CUDA: {torch.cuda.is_available()}")
print(f"Transformers: {transformers.__version__}")
print(f"PEFT: {peft.__version__}")

# 測試可選依賴
try:
    import vllm
    print(f"vLLM: {vllm.__version__}")
except ImportError:
    print("vLLM: Not installed")

try:
    import flash_attn
    print("Flash-Attention: Installed")
except ImportError:
    print("Flash-Attention: Not installed")
```

---

## 📝 已知問題

### Poetry 依賴解析錯誤
- **問題**: `Could not parse version constraint: <empty>`
- **影響**: `poetry install`, `poetry show`, `poetry update` 命令失敗
- **臨時方案**: 使用 pip 安裝可選依賴
- **長期方案**: 等待 Poetry 2.2 修復或降級到 Poetry 1.x

### 解決進度
- ✅ 已從 Poetry extras 移除 flash-attn
- ✅ 已更新 protobuf 版本約束
- ✅ 已提供 pip 安裝替代方案
- ⏸️ 等待 Poetry 官方修復

---

**維護者**: LLM 教學專案團隊
**問題追蹤**: 持續監控 Poetry 更新
