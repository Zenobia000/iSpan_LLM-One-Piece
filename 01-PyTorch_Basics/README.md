# PyTorch GPU Check

This script checks if PyTorch can successfully detect and utilize a CUDA-enabled GPU.

## Usage

To run this script using the project's Python environment, follow these steps:

1.  Navigate to the course setup directory:
    ```bash
    cd 00-Course_Setup
    ```

2.  Install the dependencies if you haven't already:
    ```bash
    poetry install
    ```

3.  Run the script using `poetry run`:
    ```bash
    poetry run python ../01-PyTorch_Basics/check_gpu.py
    ```

### Expected Output

- **If a GPU is detected:**

  The script will print a success message, the number of available GPUs, and the name of each GPU.

  ```
  --- PyTorch GPU Check ---
  Congratulations, PyTorch can access the GPU.
  Number of GPUs available: 1
  GPU 0: NVIDIA GeForce RTX 4090
  -------------------------
  ```

- **If no GPU is detected:**

  The script will print a message indicating that PyTorch could not access the GPU.

  ```
  --- PyTorch GPU Check ---
  Unfortunately, PyTorch cannot access the GPU. Please check your CUDA and cuDNN installation.
  -------------------------
  ```
