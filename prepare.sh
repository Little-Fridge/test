#!/usr/bin/env bash
set -euo pipefail

# —— 1. 加载 Conda ——  
# 如果你的 Miniconda/Anaconda 安装在其他路径，请相应修改下面这一行
source ~/miniconda3/etc/profile.d/conda.sh

# —— 2. 创建并激活 rekv 环境 ——  
conda create -n rekv python=3.11 -y
conda activate rekv

# —— 3. 升级 pip ——  
pip install --upgrade pip

# —— 4. 安装 PyTorch 及相关包 ——  
pip install -U torch torchvision torchaudio

# —— 5. 安装指定版本的 Transformers ——  
pip install -U git+https://github.com/huggingface/transformers.git@66bc4def9505fa7c7fe4aa7a248c34a026bb552b

# —— 6. 在项目根目录安装本地包 ——  
pip install -e .

# —— 7. 进入 longva 模块并安装 ——  
if [ -d "model/longva" ]; then
  pip install -e model/longva
elif [ -d "model/LongVA/longva" ]; then
  pip install -e model/LongVA/longva
else
  echo "⚠️  未检测到 longva 目录，跳过该步骤"
fi

echo "✅ prepare.sh 执行完毕，环境已就绪！"
