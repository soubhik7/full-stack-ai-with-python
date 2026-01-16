# Hugging Face Dependencies Installation Summary

## Installation Date
2026-01-16

## Python Environment

### Virtual Environment (Jupyter Kernel) ✅
**Location:** `/Users/soubhik/Desktop/AI/full-stack-ai-with-python/.venv/bin/python`  
**Kernel:** `ai_python_kernel` (AI Python Kernel 3.12)  
**Python Version:** 3.12.10  
**Status:** All dependencies installed and working ✅

## Installed Packages

### Core Hugging Face Libraries
- **transformers** (4.57.5) - State-of-the-art Natural Language Processing library
- **datasets** (4.5.0) - Easy access to datasets for ML/NLP tasks
- **huggingface-hub** (0.36.0) - Client library for the Hugging Face Hub
- **tokenizers** (0.22.2) - Fast tokenizers for NLP models

### Deep Learning Frameworks
- **torch** (2.9.1) - PyTorch deep learning framework
- **tensorflow** (2.20.0) - TensorFlow deep learning framework ✅
- **safetensors** (0.7.0) - Safe tensor serialization format

### Training & Optimization
- **accelerate** (1.12.0) - Distributed training and mixed precision
- **optuna** (4.6.0) - Hyperparameter optimization framework
- **evaluate** (0.4.6) - Model evaluation metrics
- **keras** (3.13.1) - High-level neural networks API

### Text Processing
- **sentencepiece** (0.2.1) - Unsupervised text tokenizer

### UI/Demo
- **gradio** (6.3.0) - Build ML web apps quickly

## Notebooks Supported

All notebooks in `Scikit_ML/Udemy/huggingface/` are now fully supported:
- ✅ `text_pipelines.ipynb` - Text processing pipelines
- ✅ `image_pipelines.ipynb` - Image processing pipelines  
- ✅ `image_data.ipynb` - Image dataset handling
- ✅ `fine_tuneing.ipynb` - Model fine-tuning
- ✅ `pytorch and tensorflow.ipynb` - Both PyTorch and TensorFlow sections

## Verification
All packages were successfully imported and verified working in the virtual environment!

## Installation Command Used

```bash
/Users/soubhik/Desktop/AI/full-stack-ai-with-python/.venv/bin/python -m pip install transformers datasets evaluate torch gradio huggingface-hub accelerate sentencepiece optuna tensorflow
```

## Usage

### In Jupyter Notebooks:
1. Make sure you're using the **"AI Python Kernel (3.12)"** kernel
2. If the kernel was already running, **restart it** (Kernel → Restart Kernel)
3. Run your cells - all imports should work now!

### Verify Your Kernel:
Run this in a notebook cell to verify you're using the correct environment:
```python
import sys
print(sys.executable)
# Should output: /Users/soubhik/Desktop/AI/full-stack-ai-with-python/.venv/bin/python
```

