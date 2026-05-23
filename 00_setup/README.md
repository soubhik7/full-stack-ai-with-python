# Chapter 00 — Environment Setup

> **Before you write a single line of AI code, get your environment right.**
> This takes ~15 minutes and saves hours of debugging later.

---

## Prerequisites

- macOS (these instructions; Linux is similar)
- [Homebrew](https://brew.sh) installed
- A terminal you're comfortable with

---

## Step 1 — Python 3.11

This repo requires **Python 3.11.15** (some packages aren't compatible with 3.12+ yet).

```bash
# Install via Homebrew
brew install python@3.11

# Verify
/opt/homebrew/bin/python3.11 --version
# Python 3.11.15
```

---

## Step 2 — Virtual Environment

All packages live in a single `venv/` at the repo root — one environment for everything.

```bash
cd /path/to/full-stack-ai-with-python

# Create the venv
/opt/homebrew/bin/python3.11 -m venv venv

# Activate (do this every time you open a terminal)
source venv/bin/activate

# Upgrade pip
pip install --upgrade pip

# Install all dependencies
pip install -r requirements.txt
```

You should see `(venv)` in your prompt. To deactivate: `deactivate`.

---

## Step 3 — Jupyter Kernel

Register the project venv as a named Jupyter kernel so every notebook uses the right Python.

```bash
source venv/bin/activate

pip install ipykernel

python -m ipykernel install \
  --user \
  --name KernelSoubhik \
  --display-name "KernelSoubhik"

# Verify
jupyter kernelspec list
# kernelsoubhik    /Users/<you>/Library/Jupyter/kernels/kernelsoubhik
```

---

## Step 4 — API Keys

Create a `.env` file at the repo root (it's gitignored):

```bash
cp .env.example .env 2>/dev/null || touch .env
```

Open `.env` and fill in:

```
OPENAI_API_KEY=sk-...
GOOGLE_API_KEY=...
HUGGINGFACEHUB_API_TOKEN=hf_...
LANGCHAIN_API_KEY=ls__...
GROQ_API_KEY=gsk_...
```

You don't need all keys to start — `OPENAI_API_KEY` or `GOOGLE_API_KEY` is enough for chapters 01–05.

---

## Step 5 — Docker (for AI Apps chapter)

Several apps in `08_ai_apps/` use Docker for services like Qdrant and Redis. Install Docker Desktop:

- [Docker Desktop for Mac](https://www.docker.com/products/docker-desktop/)

Verify: `docker --version`

---

## Step 6 — Launch JupyterLab

```bash
source venv/bin/activate
jupyter lab
```

In JupyterLab, when you open any notebook:
1. Click the kernel selector (top-right, shows "Python 3" or similar)
2. Select **"KernelSoubhik"**
3. All imports should resolve correctly

---

## Verification Checklist

Run this in a new notebook cell to confirm everything works:

```python
import sys
import numpy as np
import pandas as pd
import sklearn
import torch

print(f"Python: {sys.version}")
print(f"NumPy:  {np.__version__}")
print(f"Pandas: {pd.__version__}")
print(f"sklearn:{sklearn.__version__}")
print(f"PyTorch:{torch.__version__}")
print(f"Kernel: {sys.executable}")
# Should show: .../full-stack-ai-with-python/venv/bin/python
```

---

## Troubleshooting

**Kernel not visible in JupyterLab?**
```bash
source venv/bin/activate
python -m ipykernel install --user --name KernelSoubhik --display-name "KernelSoubhik"
# Restart JupyterLab
```

**`ModuleNotFoundError` in a notebook?**
```bash
source venv/bin/activate
pip install <package-name>
# Re-run the cell
```

**Wrong Python in kernel?**
```python
# In a notebook cell
import sys; print(sys.executable)
# Must show the venv path, not /usr/bin/python or /opt/homebrew/bin/python
```

---

## Next Step

Head to **[Chapter 01 — Python Fundamentals](../01_python/README.md)** →
