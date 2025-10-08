# 跨平台測試計畫與驗證清單
## Cross-Platform Testing Plan & Verification Checklist

**版本**: v1.0
**制定日期**: 2025-10-08
**適用範圍**: iSpan LLM One-Piece 完整課程專案
**測試目標**: 確保所有教學內容在主流作業系統上可執行

---

## 📋 目錄

1. [測試環境矩陣](#測試環境矩陣)
2. [測試範圍與優先級](#測試範圍與優先級)
3. [環境設置驗證](#環境設置驗證)
4. [PEFT 實驗室測試](#peft-實驗室測試)
5. [相容性問題處理](#相容性問題處理)
6. [自動化測試腳本](#自動化測試腳本)
7. [測試報告模板](#測試報告模板)

---

## 🖥️ 測試環境矩陣

### 支援的作業系統與環境

| 平台 | 版本 | GPU支援 | CPU支援 | 優先級 | 測試狀態 |
|------|------|---------|---------|--------|----------|
| **Windows 10/11** | 21H2+ | CUDA 12.1 | ✅ | 🔴 最高 | ⏳ 待測試 |
| **macOS Intel** | 12.0+ | ❌ | MPS | 🟡 中 | ⏳ 待測試 |
| **macOS Apple Silicon** | 12.0+ | ❌ | MPS/Metal | 🟡 中 | ⏳ 待測試 |
| **Ubuntu 20.04+** | 20.04/22.04 | CUDA 12.1 | ✅ | 🔴 最高 | ⏳ 待測試 |
| **WSL2 (Ubuntu)** | 20.04+ | CUDA (透傳) | ✅ | 🟢 低 | ⏳ 待測試 |

### 硬體配置需求

#### 最低配置
- **CPU**: Intel i5 / AMD Ryzen 5 或以上
- **記憶體**: 16GB RAM
- **儲存空間**: 50GB 可用空間
- **GPU**: 無 (可執行 CPU 模式)

#### 推薦配置
- **CPU**: Intel i7 / AMD Ryzen 7 或以上
- **記憶體**: 32GB RAM
- **儲存空間**: 100GB 可用空間 (SSD)
- **GPU**: NVIDIA GTX 1660 / RTX 3060 或以上 (6GB+ VRAM)

#### 理想配置
- **CPU**: Intel i9 / AMD Ryzen 9 或以上
- **記憶體**: 64GB RAM
- **儲存空間**: 500GB 可用空間 (NVMe SSD)
- **GPU**: NVIDIA RTX 4070 / RTX 4090 (12GB+ VRAM)

---

## 🎯 測試範圍與優先級

### Phase 1: 核心環境測試 (🔴 最高優先級)

#### 1.1 Poetry 環境建置
- [ ] Python 版本相容性 (3.10+)
- [ ] Poetry 安裝與初始化
- [ ] 依賴套件安裝 (pyproject.toml)
- [ ] 虛擬環境啟動
- [ ] CUDA 版本檢測

#### 1.2 核心依賴驗證
```python
# 測試腳本: test_core_dependencies.py
import torch
import transformers
import peft
import datasets
import accelerate

def test_imports():
    """測試核心套件導入"""
    print(f"✅ PyTorch: {torch.__version__}")
    print(f"✅ Transformers: {transformers.__version__}")
    print(f"✅ PEFT: {peft.__version__}")
    print(f"✅ Datasets: {datasets.__version__}")
    print(f"✅ Accelerate: {accelerate.__version__}")

def test_cuda():
    """測試 CUDA 可用性"""
    if torch.cuda.is_available():
        print(f"✅ CUDA 可用: {torch.version.cuda}")
        print(f"✅ GPU 數量: {torch.cuda.device_count()}")
        print(f"✅ GPU 名稱: {torch.cuda.get_device_name(0)}")
    else:
        print("⚠️ CUDA 不可用，將使用 CPU 模式")
```

### Phase 2: PEFT 實驗室測試 (🔴 最高優先級)

#### 測試清單

| 實驗室 | Windows | macOS Intel | macOS M1/M2 | Ubuntu | WSL2 | 備註 |
|--------|---------|-------------|-------------|--------|------|------|
| Lab-01-LoRA | ⏳ | ⏳ | ⏳ | ⏳ | ⏳ | 標竿實驗室 |
| Lab-02-AdapterLayers | ⏳ | ⏳ | ⏳ | ⏳ | ⏳ | |
| Lab-03-Prompt_Tuning | ⏳ | ⏳ | ⏳ | ⏳ | ⏳ | |
| Lab-04-Prefix_Tuning | ⏳ | ⏳ | ⏳ | ⏳ | ⏳ | |
| Lab-05-IA3 | ⏳ | ⏳ | ⏳ | ⏳ | ⏳ | 完整4階段 |
| Lab-06-BitFit | ⏳ | ⏳ | ⏳ | ⏳ | ⏳ | |
| Lab-07-P_Tuning | ⏳ | ⏳ | ⏳ | ⏳ | ⏳ | |
| Lab-08-P_Tuning_v2 | ⏳ | ⏳ | ⏳ | ⏳ | ⏳ | |

#### 每個實驗室測試項目
1. **01-Setup.ipynb**
   - [ ] 模型載入成功
   - [ ] 分詞器初始化
   - [ ] GPU/CPU 記憶體檢查
   - [ ] PEFT 配置正確

2. **02-Train.ipynb**
   - [ ] 訓練循環執行
   - [ ] 損失函數正常收斂
   - [ ] 檢查點儲存成功
   - [ ] 記憶體使用在合理範圍

3. **03-Inference.ipynb**
   - [ ] 模型載入成功
   - [ ] 推理輸出正常
   - [ ] 生成結果符合預期
   - [ ] 性能指標合理

4. **04-Merge_and_Deploy.ipynb** (若存在)
   - [ ] 模型合併成功
   - [ ] 合併後推理正常
   - [ ] 儲存格式正確

### Phase 3: 工具模組測試 (🟡 中優先級)

#### 3.1 common_utils/data_loaders.py
```python
# 測試腳本
from common_utils.data_loaders import (
    load_alpaca_dataset,
    InstructionDataset,
    InstructionDataCollator,
    quick_alpaca_loader
)

def test_data_loaders():
    # 測試 Alpaca 資料載入
    data = load_alpaca_dataset(num_samples=10)
    assert len(data) == 10, "資料載入失敗"

    # 測試快速載入器
    tokenizer, dataloader, dataset = quick_alpaca_loader(
        model_name="gpt2",
        num_samples=10
    )
    assert len(dataset) == 10, "快速載入器失敗"

    print("✅ data_loaders 測試通過")
```

#### 3.2 common_utils/model_helpers.py
```python
# 測試腳本
from common_utils.model_helpers import (
    ModelLoader,
    PEFTManager,
    SystemResourceManager
)

def test_model_helpers():
    # 測試系統資源檢測
    gpu_info = SystemResourceManager.get_gpu_info()
    memory_info = SystemResourceManager.get_memory_info()

    print(f"GPU 可用: {gpu_info['available']}")
    print(f"系統記憶體: {memory_info['total_gb']:.1f}GB")

    print("✅ model_helpers 測試通過")
```

### Phase 4: 特定平台問題驗證 (🟢 低優先級)

#### Windows 特定問題
- [ ] 路徑分隔符 (`\` vs `/`)
- [ ] 長路徑支援 (>260字元)
- [ ] 檔案權限問題
- [ ] 中文路徑支援
- [ ] CUDA 驅動版本相容性

#### macOS 特定問題
- [ ] MPS (Metal Performance Shaders) 支援
- [ ] M1/M2 ARM 架構相容性
- [ ] Rosetta 2 轉譯影響
- [ ] 記憶體共享機制
- [ ] 檔案系統大小寫敏感性

#### Linux 特定問題
- [ ] CUDA 版本與驅動對應
- [ ] 權限設定 (sudo/user)
- [ ] 環境變數配置
- [ ] 套件管理器衝突

#### WSL2 特定問題
- [ ] GPU 透傳設定
- [ ] CUDA 驅動版本
- [ ] 檔案系統性能
- [ ] 記憶體限制配置
- [ ] 與 Windows 互操作

---

## 🔧 環境設置驗證

### 自動化檢查腳本

#### check_environment.py
```python
"""
環境檢查腳本
執行: python check_environment.py
"""
import sys
import platform
import subprocess

def check_python_version():
    """檢查 Python 版本"""
    version = sys.version_info
    if version.major == 3 and version.minor >= 10:
        print(f"✅ Python {version.major}.{version.minor}.{version.micro}")
        return True
    else:
        print(f"❌ Python 版本過低: {version.major}.{version.minor}")
        return False

def check_poetry():
    """檢查 Poetry 安裝"""
    try:
        result = subprocess.run(
            ["poetry", "--version"],
            capture_output=True,
            text=True
        )
        print(f"✅ {result.stdout.strip()}")
        return True
    except FileNotFoundError:
        print("❌ Poetry 未安裝")
        return False

def check_cuda():
    """檢查 CUDA 可用性"""
    try:
        import torch
        if torch.cuda.is_available():
            print(f"✅ CUDA {torch.version.cuda} 可用")
            print(f"   GPU: {torch.cuda.get_device_name(0)}")
            print(f"   記憶體: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f}GB")
            return True
        else:
            print("⚠️ CUDA 不可用 (將使用 CPU)")
            return False
    except ImportError:
        print("❌ PyTorch 未安裝")
        return False

def check_disk_space():
    """檢查磁碟空間"""
    import shutil
    total, used, free = shutil.disk_usage("/")
    free_gb = free / (1024**3)

    if free_gb >= 50:
        print(f"✅ 磁碟空間: {free_gb:.1f}GB 可用")
        return True
    else:
        print(f"⚠️ 磁碟空間不足: {free_gb:.1f}GB 可用 (建議 >50GB)")
        return False

def main():
    print("=" * 60)
    print("環境檢查報告")
    print("=" * 60)
    print(f"作業系統: {platform.system()} {platform.release()}")
    print(f"處理器: {platform.processor()}")
    print()

    checks = [
        ("Python 版本", check_python_version()),
        ("Poetry 安裝", check_poetry()),
        ("CUDA 支援", check_cuda()),
        ("磁碟空間", check_disk_space()),
    ]

    print()
    print("=" * 60)
    passed = sum(1 for _, result in checks if result)
    total = len(checks)
    print(f"檢查結果: {passed}/{total} 通過")
    print("=" * 60)

    if passed == total:
        print("✅ 環境檢查全部通過，可以開始學習課程！")
    else:
        print("⚠️ 部分檢查未通過，請參考文件進行設置")

if __name__ == "__main__":
    main()
```

---

## 🧪 相容性問題處理

### 常見問題與解決方案

#### 問題 1: Windows CUDA 安裝失敗
**症狀**: `torch.cuda.is_available()` 返回 `False`

**解決方案**:
```bash
# 1. 檢查 NVIDIA 驅動版本
nvidia-smi

# 2. 確認 CUDA 版本
nvcc --version

# 3. 重新安裝 PyTorch (CUDA 12.1)
pip uninstall torch torchvision torchaudio
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

#### 問題 2: macOS M1/M2 相容性
**症狀**: 某些套件無法安裝 (如 `flash-attn`)

**解決方案**:
```bash
# 使用 Conda 環境
conda create -n llm-course python=3.10
conda activate llm-course

# 安裝 ARM 相容版本
conda install pytorch::pytorch torchvision torchaudio -c pytorch

# 跳過不相容套件
poetry install --no-root --without flash-attention
```

#### 問題 3: WSL2 GPU 透傳
**症狀**: WSL2 中無法使用 GPU

**解決方案**:
```bash
# 1. 確認 Windows 驅動版本 (需 >470.0)
# 在 PowerShell 執行
nvidia-smi

# 2. WSL2 中驗證
wsl
nvidia-smi

# 3. 若失敗，更新 WSL
wsl --update
```

#### 問題 4: 路徑問題 (Windows)
**症狀**: 路徑包含中文或空格導致錯誤

**解決方案**:
```python
# 使用 pathlib 處理路徑
from pathlib import Path

# 錯誤寫法
model_path = "D:\模型\llama2"  # 中文路徑問題

# 正確寫法
model_path = Path("D:/models/llama2")  # 使用正斜線
model_path = Path(r"D:\models\llama2")  # 或使用 raw string
```

---

## 🤖 自動化測試腳本

### run_lab_tests.py
```python
"""
實驗室自動化測試腳本
執行: python run_lab_tests.py --lab Lab-01-LoRA
"""
import argparse
import subprocess
import json
from pathlib import Path

def run_notebook(notebook_path):
    """執行單個 Jupyter Notebook"""
    try:
        result = subprocess.run(
            [
                "jupyter", "nbconvert",
                "--to", "notebook",
                "--execute",
                "--ExecutePreprocessor.timeout=600",
                "--output", f"{notebook_path.stem}_executed.ipynb",
                str(notebook_path)
            ],
            capture_output=True,
            text=True,
            timeout=600
        )

        if result.returncode == 0:
            print(f"✅ {notebook_path.name} 執行成功")
            return True
        else:
            print(f"❌ {notebook_path.name} 執行失敗")
            print(result.stderr)
            return False

    except subprocess.TimeoutExpired:
        print(f"⏱️ {notebook_path.name} 執行超時")
        return False
    except Exception as e:
        print(f"❌ {notebook_path.name} 執行錯誤: {e}")
        return False

def test_lab(lab_dir):
    """測試單個實驗室"""
    lab_path = Path(lab_dir)

    if not lab_path.exists():
        print(f"❌ 實驗室不存在: {lab_dir}")
        return False

    print(f"\n{'='*60}")
    print(f"測試實驗室: {lab_path.name}")
    print(f"{'='*60}\n")

    notebooks = sorted(lab_path.glob("*.ipynb"))
    results = []

    for notebook in notebooks:
        if "executed" not in notebook.name:  # 跳過已執行的輸出
            success = run_notebook(notebook)
            results.append((notebook.name, success))

    # 生成測試報告
    passed = sum(1 for _, success in results if success)
    total = len(results)

    print(f"\n{'='*60}")
    print(f"測試結果: {passed}/{total} 通過")
    print(f"{'='*60}\n")

    return passed == total

def main():
    parser = argparse.ArgumentParser(description="PEFT 實驗室自動化測試")
    parser.add_argument("--lab", required=True, help="實驗室目錄")
    args = parser.parse_args()

    success = test_lab(args.lab)
    exit(0 if success else 1)

if __name__ == "__main__":
    main()
```

---

## 📊 測試報告模板

### 測試報告範例

```markdown
# 跨平台測試報告

**測試日期**: YYYY-MM-DD
**測試環境**: Windows 11 / Ubuntu 22.04 / macOS 13
**測試人員**: [姓名]

## 環境資訊

- **作業系統**: Windows 11 Pro 22H2
- **CPU**: Intel Core i7-12700K
- **記憶體**: 32GB DDR4
- **GPU**: NVIDIA RTX 3070 (8GB)
- **Python 版本**: 3.10.11
- **CUDA 版本**: 12.1
- **PyTorch 版本**: 2.0.1

## 測試結果總覽

| 測試項目 | 狀態 | 備註 |
|---------|------|------|
| Poetry 環境建置 | ✅ | 無問題 |
| 核心依賴安裝 | ✅ | 無問題 |
| Lab-01-LoRA | ✅ | 執行時間: 15分鐘 |
| Lab-02-AdapterLayers | ✅ | 執行時間: 12分鐘 |
| Lab-03-Prompt_Tuning | ⚠️ | 記憶體不足警告 |
| common_utils 測試 | ✅ | 無問題 |

## 詳細測試記錄

### Lab-01-LoRA
- **01-Setup.ipynb**: ✅ 通過 (執行時間: 2分鐘)
- **02-Train.ipynb**: ✅ 通過 (執行時間: 10分鐘)
- **03-Inference.ipynb**: ✅ 通過 (執行時間: 2分鐘)
- **04-Merge_and_Deploy.ipynb**: ✅ 通過 (執行時間: 1分鐘)

### 遇到的問題

#### 問題 1: Lab-03 記憶體警告
**描述**: 訓練過程中出現記憶體不足警告
**解決方案**: 降低批次大小從 4 到 2
**狀態**: ✅ 已解決

## 建議與改進

1. 在 README 中補充記憶體需求說明
2. 提供批次大小調整指引
3. 補充 CPU 模式執行說明

## 簽核

- **測試人員**: ____________
- **審核人員**: ____________
- **日期**: YYYY-MM-DD
```

---

## ✅ 執行檢核清單

### 測試前準備
- [ ] 備份當前環境配置
- [ ] 準備乾淨的測試環境
- [ ] 下載必要的測試資料
- [ ] 準備測試報告模板

### 測試執行
- [ ] 執行 `check_environment.py`
- [ ] 測試 Poetry 環境建置
- [ ] 逐個測試 PEFT 實驗室
- [ ] 測試 common_utils 模組
- [ ] 記錄所有錯誤與警告

### 測試後處理
- [ ] 整理測試日誌
- [ ] 撰寫測試報告
- [ ] 提交問題與改進建議
- [ ] 更新相容性文件

---

## 📅 測試時程規劃

### Week 1: Windows 平台測試
- Day 1-2: 環境設置與核心依賴驗證
- Day 3-4: PEFT 實驗室測試 (Lab-01 到 Lab-04)
- Day 5: PEFT 實驗室測試 (Lab-05 到 Lab-08)

### Week 2: Linux 平台測試
- Day 1-2: Ubuntu 環境測試
- Day 3: WSL2 環境測試
- Day 4-5: 問題修復與驗證

### Week 3: macOS 平台測試
- Day 1-2: macOS Intel 測試
- Day 3-4: macOS M1/M2 測試
- Day 5: 綜合報告撰寫

---

**文件版本**: v1.0
**最後更新**: 2025-10-08
**維護者**: 跨平台測試小組
**審核狀態**: ✅ 已批准

**變更日誌**:
- 2025-10-08: 初始版本發布，建立完整測試框架
