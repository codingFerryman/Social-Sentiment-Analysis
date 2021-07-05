#!/bin/bash

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"

# If you want you can also add it to your ~/.bashrc
module load gcc/6.3.0 python_gpu/3.8.5 hdf5/1.10.1

python -m venv --system-site-packages "${SCRIPT_DIR}"/venv

# If you want you can also add it to your ~/.bashrc
source "${SCRIPT_DIR}"/venv/bin/activate

pip install --upgrade pip setuptools wheel
pip install torch==1.8.1+cu102 -f https://download.pytorch.org/whl/lts/1.8/torch_lts.html
pip install 'torchtext<0.10'
pip install --upgrade 'spacy[cuda102]'
spacy download en_core_web_sm
#spacy download en_core_web_trf
pip install -r "${SCRIPT_DIR}"/requirements.txt
#python "${SCRIPT_DIR}"/setup.py install
