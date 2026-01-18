# Setup & Configuration Guide
## Advanced Deep Learning Project

### 📋 Prerequisites

#### System Requirements
- **RAM**: Minimum 8GB (16GB+ recommended)
- **GPU**: Optional but highly recommended
  - NVIDIA GPU with CUDA support (recommended)
  - Apple Silicon (Metal support)
  - CPU mode (slow but works)
- **Disk Space**: 2GB minimum (for models and data)
- **Python Version**: 3.8+

#### Supported Platforms
- ✅ macOS (Intel & Apple Silicon)
- ✅ Linux (Ubuntu, Debian, etc.)
- ✅ Windows (WSL2 recommended)
- ✅ Google Colab
- ✅ Kaggle Notebooks

---

## 🔧 Installation Guide

### Option 1: Local Installation (Recommended)

#### Step 1: Create Virtual Environment
```bash
# Using conda (recommended)
conda create -n advanced_dl python=3.10
conda activate advanced_dl

# Or using venv
python3 -m venv advanced_dl_env
source advanced_dl_env/bin/activate  # macOS/Linux
# or
.\advanced_dl_env\Scripts\activate   # Windows
```

#### Step 2: Install PyTorch
```bash
# For macOS (with Metal support)
pip install torch torchvision torchaudio

# For Linux with CUDA 11.8
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# For CPU only
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
```

#### Step 3: Install Project Dependencies
```bash
pip install transformers datasets matplotlib seaborn numpy scikit-learn tqdm

# For Jupyter support
pip install jupyter jupyterlab ipykernel
```

#### Step 4: Verify Installation
```python
python -c "
import torch
import transformers
import matplotlib
import numpy
print('✓ All packages installed successfully')
print(f'PyTorch version: {torch.__version__}')
print(f'GPU available: {torch.cuda.is_available()}')
print(f'Device: {torch.device(\"cuda\" if torch.cuda.is_available() else \"cpu\")}')
"
```

### Option 2: Google Colab (Fastest)

```python
# Run in first cell of Colab notebook

# Install packages
!pip install transformers datasets matplotlib seaborn

# Verify
import torch
print(f"GPU available: {torch.cuda.is_available()}")
print(f"Device: {torch.device('cuda' if torch.cuda.is_available() else 'cpu')}")
```

### Option 3: Docker (Most Reproducible)

```dockerfile
FROM pytorch/pytorch:2.0-cuda11.8-runtime-ubuntu22.04

RUN apt-get update && apt-get install -y \
    git \
    wget \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir \
    transformers \
    datasets \
    matplotlib \
    seaborn \
    jupyter

WORKDIR /workspace
EXPOSE 8888
CMD ["jupyter", "notebook", "--ip=0.0.0.0", "--no-browser"]
```

Run with:
```bash
docker build -t advanced-dl .
docker run --gpus all -p 8888:8888 -v $(pwd):/workspace advanced-dl
```

---

## ⚙️ Configuration

### GPU Setup

#### NVIDIA GPU (CUDA)
```bash
# Check CUDA availability
nvidia-smi

# Expected output:
# +-----------------------------------------------------------------------------+
# | NVIDIA-SMI 525.105.02   Driver Version: 525.105.02 CUDA Version: 12.0     |
# +-----------------------------------------------------------------------------+

# Verify PyTorch sees GPU
python -c "import torch; print(torch.cuda.is_available())"
```

#### Apple Silicon (Metal)
```bash
# Check Metal support
python -c "import torch; print(torch.backends.mps.is_available())"

# Expected: True
```

#### CPU Only
```bash
# No special setup needed, will be slower
# Expect 10-50x slower training compared to GPU
```

### Environment Variables

```bash
# Optional: Set number of threads
export OMP_NUM_THREADS=8
export CUDA_VISIBLE_DEVICES=0  # Use GPU 0

# Optional: For reproducibility
export PYTHONHASHSEED=42

# Optional: Increase memory
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb=512
```

### Jupyter Configuration

Create `~/.jupyter/jupyter_notebook_config.py`:
```python
c.NotebookApp.open_browser = False
c.NotebookApp.port = 8888
c.NotebookApp.ip = 'localhost'
c.NotebookApp.notebook_dir = '/path/to/projects'
```

---

## 🚀 Running the Project

### Method 1: Jupyter Notebook
```bash
cd 05_Final_project
jupyter notebook advanced_deep_learning_project.ipynb
```

Browser will open at `http://localhost:8888`

### Method 2: Jupyter Lab (Modern Interface)
```bash
cd 05_Final_project
jupyter lab advanced_deep_learning_project.ipynb
```

### Method 3: VS Code
```bash
# Install Python extension
# Open notebook in VS Code
code 05_Final_project/advanced_deep_learning_project.ipynb
```

### Method 4: Command Line (Non-interactive)
```bash
jupyter nbconvert --to notebook --execute --ExecutePreprocessor.timeout=3600 \
    advanced_deep_learning_project.ipynb \
    --output advanced_deep_learning_project_executed.ipynb
```

---

## 📊 Hardware Recommendations

### Minimum (Will work slowly)
- CPU: Quad-core processor
- RAM: 8GB
- Storage: 2GB SSD
- Estimated time: 2-3 hours

### Recommended (Comfortable)
- CPU: 8-core processor
- RAM: 16GB
- GPU: 6GB VRAM (RTX 3060, T4, etc.)
- Storage: 10GB SSD
- Estimated time: 30-50 minutes

### Optimal (Best experience)
- CPU: 16+ core processor
- RAM: 32GB+
- GPU: 24GB VRAM (RTX 3090, A100, etc.)
- Storage: 20GB+ SSD (NVMe)
- Estimated time: 10-15 minutes

---

## 🔍 Troubleshooting Setup

### Issue: "ModuleNotFoundError: No module named 'torch'"
```bash
# Solution: Reinstall PyTorch
pip uninstall torch
pip install torch
```

### Issue: "CUDA out of memory"
```bash
# Solution 1: Reduce batch size in notebook
batch_size = 4  # Was 8

# Solution 2: Clear cache
import torch
torch.cuda.empty_cache()

# Solution 3: Use CPU
device = 'cpu'
```

### Issue: "No space left on device"
```bash
# Check disk space
df -h

# Clear pip cache
pip cache purge

# Clear PyTorch cache
rm -rf ~/.cache/huggingface/
```

### Issue: "Jupyter token error"
```bash
# Generate new token
jupyter notebook --generate-config
jupyter notebook password

# Or disable authentication
jupyter notebook --NotebookApp.token='' --NotebookApp.password=''
```

### Issue: "ImportError: libcuda.so.1 not found"
```bash
# Solution: CUDA not installed or not in PATH
# For Ubuntu:
sudo apt-get install nvidia-driver-XXX cuda-toolkit-11.8

# For macOS: Use Metal instead (automatic with correct PyTorch install)
```

---

## 📈 Performance Tips

### Optimize Training Speed

```python
# 1. Use GPU
device = 'cuda'

# 2. Increase batch size (if memory allows)
batch_size = 16

# 3. Use mixed precision training
from torch.cuda.amp import autocast, GradScaler
scaler = GradScaler()

# 4. Use efficient implementations
import torch.backends.cudnn as cudnn
cudnn.benchmark = True

# 5. Parallel computation
if torch.cuda.device_count() > 1:
    model = torch.nn.DataParallel(model)
```

### Monitor Performance

```bash
# Watch GPU usage in real-time
watch -n 1 nvidia-smi

# Or use alternative
gpustat --watch

# Monitor CPU and memory
top
```

### Profile Your Code

```python
import cProfile
import pstats

profiler = cProfile.Profile()
profiler.enable()

# ... your training code ...

profiler.disable()
stats = pstats.Stats(profiler)
stats.sort_stats('cumulative')
stats.print_stats(10)
```

---

## 🔐 Security Best Practices

### API Keys
```python
# Never hardcode tokens!
import os
from dotenv import load_dotenv

load_dotenv()
hf_token = os.getenv('HF_TOKEN')
```

### Save Credentials Safely
```bash
# Create .env file (add to .gitignore!)
echo "HF_TOKEN=your_token_here" > .env

# In Python
from huggingface_hub import login
login()  # Uses token from ~/.cache/huggingface/token
```

### Protect Sensitive Data
```bash
# Add to .gitignore
echo ".env" >> .gitignore
echo "*.pt" >> .gitignore
echo "model_*/" >> .gitignore
echo ".ipynb_checkpoints/" >> .gitignore
```

---

## 📦 Dependency Versions

**Tested with:**
```
Python: 3.10.0+
torch: 2.0.0+
transformers: 4.30.0+
datasets: 2.12.0+
matplotlib: 3.7.0+
seaborn: 0.12.0+
numpy: 1.24.0+
scikit-learn: 1.3.0+
jupyter: 1.0.0+
```

**Version Constraints:**
```
torch>=2.0.0              # Stable API
transformers>=4.20.0     # For HF Hub
datasets>=2.0.0          # For data loading
matplotlib>=3.5.0        # For plotting
```

---

## 🎯 Verification Checklist

Before running the notebook, verify:

- [ ] Python 3.8+ installed (`python --version`)
- [ ] Virtual environment activated
- [ ] PyTorch installed correctly (`import torch`)
- [ ] GPU detected (if applicable) (`torch.cuda.is_available()`)
- [ ] All dependencies installed (`pip list | grep torch`)
- [ ] Jupyter running (`jupyter --version`)
- [ ] Disk space available (`df -h`)
- [ ] RAM available (`free -h` or Activity Monitor)
- [ ] Internet connection (for model downloads)
- [ ] Folder structure correct

---

## 📚 Additional Resources

### Official Documentation
- [PyTorch](https://pytorch.org/docs/stable/index.html)
- [Transformers](https://huggingface.co/docs/transformers/)
- [Jupyter](https://jupyter-notebook.readthedocs.io/)

### Troubleshooting Resources
- [PyTorch Discussions](https://discuss.pytorch.org/)
- [Hugging Face Issues](https://github.com/huggingface/transformers/issues)
- [Stack Overflow](https://stackoverflow.com/questions/tagged/pytorch)

---

## ✅ Next Steps

Once setup is complete:

1. ✅ Verify installation (see Troubleshooting Checklist)
2. ✅ Launch Jupyter notebook
3. ✅ Run first cell (imports and setup)
4. ✅ Follow notebook sections in order
5. ✅ Save checkpoints as you go
6. ✅ Review generated visualizations

**Ready to start?** → Open `advanced_deep_learning_project.ipynb`

---

**Last Updated**: January 2026
**Support**: Check README.md and QUICK_REFERENCE.md for more help
