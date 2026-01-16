# Quick Start Guide - Hugging Face Notebooks

## ✅ Setup Complete!

All dependencies have been installed in your project's **virtual environment** (`.venv`).

## 🚀 How to Use

### 1. Open Your Notebook
Open any notebook in `Scikit_ML/Udemy/huggingface/` in Jupyter.

### 2. Verify/Select the Correct Kernel
Your notebook should already be using the **"AI Python Kernel (3.12)"** kernel.

**To verify or change the kernel:**
1. Look at the top-right corner of your notebook
2. It should show **"AI Python Kernel (3.12)"**
3. If it shows a different kernel, click on it and select **"AI Python Kernel (3.12)"**

### 3. Restart the Kernel (Important!)
Since we just installed new packages:
1. Go to **Kernel** menu → **Restart Kernel**
2. Click **Restart** to confirm

### 4. Run Your Code
Try running your first cell again. The `ModuleNotFoundError` should be gone! 🎉

## 📦 Installed Packages

| Package | Version | Purpose |
|---------|---------|---------|
| transformers | 4.57.5 | NLP models |
| datasets | 4.5.0 | Dataset management |
| torch | 2.9.1 | PyTorch framework |
| tensorflow | 2.20.0 | TensorFlow framework |
| gradio | 6.3.0 | Build ML demos |
| optuna | 4.6.0 | Hyperparameter tuning |
| accelerate | 1.12.0 | Distributed training |
| keras | 3.13.1 | Neural networks API |

## 🔧 Troubleshooting

### If you still get "ModuleNotFoundError"
1. **Check your kernel**: Make sure you're using **"AI Python Kernel (3.12)"**
2. **Restart the kernel**: Kernel → Restart Kernel (this is crucial!)
3. **Verify installation**:
   ```python
   import sys
   print(sys.executable)  
   # Should show: /Users/soubhik/Desktop/AI/full-stack-ai-with-python/.venv/bin/python
   ```

### If you need to reinstall
```bash
/Users/soubhik/Desktop/AI/full-stack-ai-with-python/.venv/bin/python -m pip install --upgrade transformers datasets evaluate torch gradio huggingface-hub accelerate sentencepiece optuna tensorflow
```

## 💡 Tips

- **First time running?** The first import of transformers may take a moment as it migrates cache
- **GPU support:** PyTorch will automatically use your Mac's GPU (MPS) if available
- **Model downloads:** Models will be cached in `~/.cache/huggingface/`

## 📚 Resources

- [Transformers Documentation](https://huggingface.co/docs/transformers)
- [Datasets Documentation](https://huggingface.co/docs/datasets)
- [Hugging Face Hub](https://huggingface.co/models)

---

**Ready to go!** Open any notebook and start experimenting with Hugging Face! 🎉
