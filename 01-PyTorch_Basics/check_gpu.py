import torch

def check_gpu():
    """
    Checks for the availability of CUDA-enabled GPUs using PyTorch.

    This function prints information about the availability of GPUs,
    the number of available GPUs, and their names.
    """
    print("--- PyTorch GPU Check ---")
    if torch.cuda.is_available():
        print("Congratulations, PyTorch can access the GPU.")
        print(f"Number of GPUs available: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
    else:
        print("Unfortunately, PyTorch cannot access the GPU. Please check your CUDA and cuDNN installation.")
    print("-------------------------")

if __name__ == "__main__":
    check_gpu()
