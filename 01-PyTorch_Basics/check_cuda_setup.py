#!/usr/bin/env python3
"""
CUDA 設定檢查腳本
Check CUDA setup and PyTorch installation compatibility
"""

import sys
import subprocess
import platform

def check_system_info():
    """檢查系統資訊"""
    print("=== 系統資訊 ===")
    print(f"作業系統: {platform.system()} {platform.release()}")
    print(f"Python 版本: {sys.version}")
    print()

def check_cuda_driver():
    """檢查 CUDA Driver"""
    print("=== CUDA Driver 檢查 ===")
    try:
        result = subprocess.run(['nvidia-smi'], capture_output=True, text=True)
        if result.returncode == 0:
            print("✅ NVIDIA Driver 正常運作")
            # 提取 CUDA 版本
            lines = result.stdout.split('\n')
            for line in lines:
                if 'CUDA Version:' in line:
                    cuda_version = line.split('CUDA Version:')[1].strip().split()[0]
                    print(f"Driver 支援的 CUDA 版本: {cuda_version}")
            print()
        else:
            print("❌ NVIDIA Driver 未正確安裝或運作")
            return False
    except FileNotFoundError:
        print("❌ nvidia-smi 命令未找到，請檢查 NVIDIA Driver 安裝")
        return False

    return True

def check_cuda_toolkit():
    """檢查 CUDA Toolkit"""
    print("=== CUDA Toolkit 檢查 ===")
    try:
        result = subprocess.run(['nvcc', '--version'], capture_output=True, text=True)
        if result.returncode == 0:
            print("✅ CUDA Toolkit 已安裝")
            # 提取 CUDA 版本
            lines = result.stdout.split('\n')
            for line in lines:
                if 'release' in line:
                    version = line.split('release')[1].split(',')[0].strip()
                    print(f"CUDA Toolkit 版本: {version}")
            print()
            return True
        else:
            print("❌ CUDA Toolkit 未正確安裝")
            return False
    except FileNotFoundError:
        print("❌ nvcc 命令未找到，請檢查 CUDA Toolkit 安裝")
        return False

def check_pytorch():
    """檢查 PyTorch 安裝"""
    print("=== PyTorch 檢查 ===")
    try:
        import torch
        print(f"✅ PyTorch 版本: {torch.__version__}")

        # 檢查 CUDA 可用性
        if torch.cuda.is_available():
            print(f"✅ CUDA 在 PyTorch 中可用")
            print(f"CUDA 版本: {torch.version.cuda}")
            print(f"CuDNN 版本: {torch.backends.cudnn.version()}")
            print(f"GPU 數量: {torch.cuda.device_count()}")

            # 列出所有 GPU
            for i in range(torch.cuda.device_count()):
                gpu_name = torch.cuda.get_device_name(i)
                gpu_memory = torch.cuda.get_device_properties(i).total_memory / 1024**3
                print(f"GPU {i}: {gpu_name} ({gpu_memory:.1f} GB)")
        else:
            print("❌ CUDA 在 PyTorch 中不可用")
            return False

        print()
        return True

    except ImportError:
        print("❌ PyTorch 未安裝")
        return False

def check_other_libraries():
    """檢查其他重要套件"""
    print("=== 其他套件檢查 ===")

    libraries = [
        'transformers',
        'datasets',
        'accelerate',
        'peft',
        'bitsandbytes',
        'flash_attn'
    ]

    results = {}

    for lib in libraries:
        try:
            if lib == 'flash_attn':
                import flash_attn
                version = flash_attn.__version__
            else:
                module = __import__(lib)
                version = getattr(module, '__version__', 'Unknown')

            print(f"✅ {lib}: {version}")
            results[lib] = True
        except ImportError:
            print(f"❌ {lib}: 未安裝")
            results[lib] = False

    print()
    return results

def test_cuda_operations():
    """測試 CUDA 操作"""
    print("=== CUDA 操作測試 ===")

    try:
        import torch

        if not torch.cuda.is_available():
            print("❌ CUDA 不可用，跳過測試")
            return False

        # 基本張量操作
        device = torch.device('cuda')
        print(f"使用設備: {device}")

        # 創建張量並移到 GPU
        x = torch.randn(1000, 1000).to(device)
        y = torch.randn(1000, 1000).to(device)

        # 矩陣乘法
        import time
        start_time = time.time()
        z = torch.mm(x, y)
        torch.cuda.synchronize()  # 等待 GPU 完成
        end_time = time.time()

        print(f"✅ GPU 矩陣乘法成功")
        print(f"計算時間: {end_time - start_time:.4f} 秒")

        # 記憶體資訊
        memory_allocated = torch.cuda.memory_allocated() / 1024**2
        memory_reserved = torch.cuda.memory_reserved() / 1024**2
        print(f"GPU 記憶體使用: {memory_allocated:.1f} MB 已分配, {memory_reserved:.1f} MB 已保留")

        print()
        return True

    except Exception as e:
        print(f"❌ CUDA 操作測試失敗: {e}")
        return False

def main():
    """主函式"""
    print("CUDA 環境檢查工具")
    print("=" * 50)

    # 檢查各項目
    system_ok = True
    cuda_driver_ok = check_cuda_driver()
    cuda_toolkit_ok = check_cuda_toolkit()
    pytorch_ok = check_pytorch()
    libraries = check_other_libraries()
    cuda_ops_ok = test_cuda_operations()

    # 總結
    print("=== 檢查總結 ===")

    if cuda_driver_ok and cuda_toolkit_ok and pytorch_ok and cuda_ops_ok:
        print("🎉 所有檢查通過！CUDA 環境設定正確")

        # 提供建議
        print("\n=== 建議 ===")
        if not libraries.get('flash_attn', False):
            print("💡 建議安裝 Flash Attention 以提升訓練效率：")
            print("   pip install flash-attn --no-build-isolation")

        if not libraries.get('bitsandbytes', False):
            print("💡 建議安裝 BitsAndBytes 以支援量化：")
            print("   pip install bitsandbytes")

    else:
        print("❌ 部分檢查未通過，請檢查上述問題")

        if not cuda_driver_ok:
            print("🔧 請安裝或更新 NVIDIA GPU Driver")

        if not cuda_toolkit_ok:
            print("🔧 請安裝 CUDA Toolkit 11.5")

        if not pytorch_ok:
            print("🔧 請安裝支援 CUDA 11.5 的 PyTorch：")
            print("   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu115")

if __name__ == "__main__":
    check_system_info()
    main()