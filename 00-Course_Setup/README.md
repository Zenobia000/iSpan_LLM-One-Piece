# 00-Course_Setup: Environment and Project Initialization

This directory contains the necessary files to set up the Python environment for the LLM Engineering course using [Poetry](https://python-poetry.org/).

## Why Poetry?

Poetry is a modern tool for Python dependency management and packaging. It provides a deterministic build process by using a `poetry.lock` file, ensuring that everyone working on the project has the exact same versions of all dependencies. This is crucial for reproducibility in machine learning projects.

## Setup Instructions

### 1. Install Poetry

First, install Poetry on your system. Please choose the instructions corresponding to your operating system.

<details>
<summary><strong>macOS / Linux</strong></summary>

```bash
curl -sSL https://install.python-poetry.org | python3 -
```
</details>

<details>
<summary><strong>WSL (Windows Subsystem for Linux) - Recommended Environment</strong></summary>

This is the environment used for the development of this course.

```bash
curl -sSL https://install.python-poetry.org | python3 -
```
</details>

<details>
<summary><strong>Windows (PowerShell)</strong></summary>

```powershell
(Invoke-WebRequest -Uri https://install.python-poetry.org -UseBasicParsing).Content | py -
```
</details>

#### **1.1. Configure PATH Environment Variable (For macOS, Linux, WSL)**

After installation, you need to add Poetry's bin directory to your shell's `PATH` to use the `poetry` command directly.

- If you use **bash** (common on Linux/WSL), run this command:
  ```bash
  echo 'export PATH="$HOME/.local/bin:$PATH"' >> ~/.bashrc
  source ~/.bashrc
  ```
- If you use **zsh** (default on modern macOS), run this command:
  ```bash
  echo 'export PATH="$HOME/.local/bin:$PATH"' >> ~/.zshrc
  source ~/.zshrc
  ```
On **Windows**, the installer should handle this automatically. If not, you'll need to add the path to your system's Environment Variables manually.

### 2. Configure Poetry to Create Virtual Environments in the Project Directory

For easier management, it's recommended to configure Poetry to create the virtual environment inside the project folder. This creates a `.venv` directory that is easy to find and manage. This command is the same for all platforms.

```bash
poetry config virtualenvs.in-project true
```

### 3. Install Dependencies

Navigate to this directory (`00-Course_Setup/`) in your terminal and run the following command. This command is the same for all platforms.

```bash
poetry install --no-root --all-extras
```
This command will:
1.  Read the `pyproject.toml` file.
2.  Create a new virtual environment inside the `./.venv` directory if it doesn't exist.
3.  Install all the dependencies specified in the file, including the optional `inference` group.

> **Note:** The `--no-root` flag is important because this directory is used only for dependency management and is not a runnable Python package itself.

### 4. Activate the Virtual Environment

To use the installed packages and run your scripts, you need to activate the virtual environment.

#### **a) Using `poetry env activate` (Recommended for Poetry >= 2.0)**
Poetry 2.0+ provides a replacement for the `shell` command. This method shows you the command to activate the virtual environment:

```bash
poetry env activate
```

This will output something like:
```bash
source /path/to/your/virtualenv/bin/activate
```

Copy and run the displayed command to activate the virtual environment.

#### **b) Alternative: Manual virtual environment activation**
If you prefer to activate directly, you can use the path shown by `poetry env info`:

```bash
# First, get the virtual environment path
poetry env info --path

# Then activate using the path (example for Linux/macOS/WSL)
source $(poetry env info --path)/bin/activate
```

> **Note on poetry-plugin-shell:** While the `poetry shell` command can be restored with `poetry self add poetry-plugin-shell`, this may cause dependency conflicts on some systems. The `poetry env activate` method is more reliable.

You can now run Python scripts (e.g., `python your_script.py`) or start Jupyter Lab (`jupyter lab`).

#### **c) Using the `activate` script directly (Alternative method)**
If you prefer to activate it in your current shell without using Poetry commands, use the command for your specific OS and shell.

<details>
<summary><strong>macOS / Linux / WSL (bash/zsh)</strong></summary>

```bash
source ./.venv/bin/activate
```
</details>

<details>
<summary><strong>Windows (Command Prompt)</strong></summary>

```bash
.\.venv\Scripts\activate
```
</details>

<details>
<summary><strong>Windows (PowerShell)</strong></summary>

```powershell
.\.venv\Scripts\Activate.ps1
```
</details>

### Important Notes on GPU Dependencies

Some libraries like `TensorRT`, `vLLM`, and `bitsandbytes` have specific CUDA version requirements. The `pyproject.toml` file specifies general versions, but you might need to install a specific version that matches your NVIDIA driver and CUDA toolkit.

Please consult the official documentation for these packages if you encounter installation issues.
