# Centralized Environment Setup

This project now uses a **unified virtual environment** and **single Jupyter kernel** for all Python work across the entire codebase.

## Virtual Environment

### Location
- **Path**: `/Users/soubhik/AI/full-stack-ai-with-python/.venv`
- **Python Version**: Python 3.14

### Activation

To activate the virtual environment:

```bash
cd /Users/soubhik/AI/full-stack-ai-with-python
source .venv/bin/activate
```

You should see `(.venv)` in your terminal prompt.

### Deactivation

To deactivate:

```bash
deactivate
```

---

## Jupyter Kernel

### Kernel Name
- **Kernel Name**: `KernelSoubhik01`
- **Display Name**: "KernelSoubhik01"

### Using the Kernel

1. **In Jupyter Notebooks**:
   - Open any `.ipynb` file in the project
   - Click on the kernel selector (usually top-right corner)
   - Select **"KernelSoubhik01"**

2. **Verify Kernel**:
   ```bash
   jupyter kernelspec list
   ```
   You should see `kernelsoubhik01` in the list.

### Available Kernels

After cleanup, only the following kernels are available:
- `kernelsoubhik01` - **Use this for all Python notebooks**
- `bash` - For Bash notebooks
- `javascript`, `jslab`, `tslab` - For JavaScript/TypeScript notebooks
- `python3` - System Python (avoid using this)

---

## Dependency Management

### Adding New Packages

1. **Add to `requirements.txt`**:
   ```bash
   echo "package-name==version" >> requirements.txt
   ```

2. **Install the package**:
   ```bash
   source .venv/bin/activate
   pip install package-name
   ```

3. **Update `requirements.txt` with exact versions**:
   ```bash
   pip freeze > requirements.txt
   ```

### Reinstalling All Dependencies

If you need to reinstall everything:

```bash
cd /Users/soubhik/AI/full-stack-ai-with-python
rm -rf .venv
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip setuptools wheel
pip install -r requirements.txt
```

---

## Key Packages Installed

The unified environment includes:

### Machine Learning & Data Science
- `numpy==2.3.5`
- `pandas==2.3.3`
- `scikit-learn==1.8.0`
- `scipy==1.17.0`
- `matplotlib==3.10.8`
- `seaborn==0.13.2`

### Deep Learning & NLP
- `torch==2.9.1`
- `transformers==4.57.5`
- `datasets==4.5.0`
- `spacy==3.8.11`
- `sentence-transformers`

### LangChain & AI
- `langchain-classic==1.0.0`
- `langchain-core==1.1.0`
- `langchain-community==0.4.1`
- `langchain-openai==1.1.0`
- `langchain-huggingface`
- `openai==2.8.1`

### Jupyter & Development
- `jupyterlab==4.5.2`
- `ipykernel==7.1.0`
- `notebook==7.5.2`

### Other Tools
- `gradio==6.3.0`
- `optuna==4.6.0`
- `faiss-cpu==1.7.4`
- `qdrant-client==1.16.1`

---

## Troubleshooting

### Kernel Not Found in Jupyter

If you don't see the kernel in Jupyter:

1. **Verify kernel is installed**:
   ```bash
   jupyter kernelspec list
   ```

2. **Reinstall the kernel**:
   ```bash
   source .venv/bin/activate
   python -m ipykernel install --user --name=KernelSoubhik01 --display-name="KernelSoubhik01"
   ```

3. **Restart Jupyter**:
   - Close all Jupyter instances
   - Restart Jupyter Lab/Notebook

### Import Errors

If you get `ModuleNotFoundError`:

1. **Check if package is installed**:
   ```bash
   source .venv/bin/activate
   pip list | grep package-name
   ```

2. **Install missing package**:
   ```bash
   pip install package-name
   ```

3. **Verify kernel is using correct Python**:
   In a notebook cell, run:
   ```python
   import sys
   print(sys.executable)
   ```
   Should show: `/Users/soubhik/AI/full-stack-ai-with-python/.venv/bin/python`

### Dependency Conflicts

If you encounter version conflicts:

1. **Check conflicting packages**:
   ```bash
   source .venv/bin/activate
   pip check
   ```

2. **Review `requirements.txt`** for duplicate entries

3. **Remove duplicates** keeping the latest compatible version

---

## Migration Notes

### What Changed

1. **Removed Kernels** (10 duplicates removed):
   - `ai_dnd`, `ai_python_kernel`, `full-stack-ai-venv`
   - `full-stack-ai-with-python-env`, `fullstack-ai`, `fullstack-ai-3.13`
   - `myenv`, `project2-venv`, `python3126`, `venv`

2. **Cleaned `requirements.txt`**:
   - Removed duplicate package entries
   - Resolved version conflicts (kept latest versions)
   - Removed incompatible old `langchain==0.0.297`

### Migrating Existing Notebooks

For any existing notebooks:

1. Open the notebook
2. Select **"KernelSoubhik01"** kernel
3. Run cells to verify functionality
4. If imports fail, check [Troubleshooting](#troubleshooting) section

---

## Best Practices

1. **Always activate the virtual environment** before installing packages
2. **Use the unified kernel** for all Python notebooks
3. **Document new dependencies** in `requirements.txt`
4. **Test notebooks** after adding new dependencies
5. **Don't modify `.venv` manually** - use `pip` to manage packages
