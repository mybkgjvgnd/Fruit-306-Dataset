# 🚀 Quick Start Guide

This guide provides detailed instructions on how to use the Fruit-306 dataset and FruitEnsemble framework.

### 📋 Prerequisites

#### System Requirements
- **OS**: Linux (Ubuntu 18.04+), Windows 10/11, or macOS 12+
- **Python**: 3.8 or higher
- **GPU**: NVIDIA GPU with 8GB+ VRAM (recommended for training)
- **RAM**: 16GB+ recommended
- **Storage**: 30GB+ free space for dataset and models

#### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/Fruit-306
   cd Fruit-306
   ```

# **Create virtual environment** (recommended)

```bash
# Using conda
conda create -n fruit306 python=3.9
conda activate fruit306

# Or using venv
python -m venv fruit306_env
source fruit306_env/bin/activate  # Linux/Mac
# fruit306_env\Scripts\activate  # Windows
```

**Install dependencies**

```bash
pip install -r requirements.txt
```

**requirements.txt** content:

```txt
# Core dependencies
torch>=1.9.0
torchvision>=0.10.0
numpy>=1.19.5
pandas>=1.3.3
Pillow>=8.3.1

# Utilities
tqdm>=4.62.3
scikit-learn>=0.24.2
matplotlib>=3.4.3

# LLM API
dashscope>=1.10.0  # For Qwen-VL

# Optional
opencv-python>=4.5.3
seaborn>=0.11.2
jupyter>=1.0.0
```

**Set up API key** (for LLM arbitration)

```bash
# Linux/Mac
export DASHSCOPE_API_KEY="your-api-key-here"

# Windows (Command Prompt)
set DASHSCOPE_API_KEY=your-api-key-here

# Windows (PowerShell)
$env:DASHSCOPE_API_KEY="your-api-key-here"
```

