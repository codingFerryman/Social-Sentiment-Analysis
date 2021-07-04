#!/bin/bash

# This script is for setting up the environment on the local machine
# You should have python3.8-venv installed on your computer

CIL_LOCALREPO=$(cd "$(dirname "${BASH_SOURCE[0]}")" &>/dev/null && pwd)

chmod +x ${CIL_LOCALREPO}/setup_dataset.sh
chmod +x ${CIL_LOCALREPO}/setup_environment.sh

${CIL_LOCALREPO}/setup_dataset.sh

python -m venv --system-site-packages "${CIL_LOCALREPO}"/venv
source "${CIL_LOCALREPO}"/venv/bin/activate
pip install --upgrade pip setuptools wheel

# If you have CUDA installed on your computer
#pip install torch==1.8.1+cu102 -f https://download.pytorch.org/whl/lts/1.8/torch_lts.html
#pip install --upgrade 'spacy[cuda102]'

# If you DON'T have CUDA installed on your computer
pip install torch==1.8.1+cpu -f https://download.pytorch.org/whl/lts/1.8/torch_lts.html
pip install --upgrade 'spacy' # If you DON'T have CUDA installed on your computer

pip install 'torchtext<0.10'
spacy download en_core_web_sm
#spacy download en_core_web_trf
pip install -r "${CIL_LOCALREPO}"/requirements.txt
#python "${SCRIPT_DIR}"/setup.py install
