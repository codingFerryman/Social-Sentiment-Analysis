#!/bin/bash

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"

# If you want you can also add it to your ~/.bashrc
module load gcc/6.3.0 python_gpu/3.8.5 hdf5/1.10.1 eth_proxy

python -m venv "${SCRIPT_DIR}"/venv

# If you want you can also add it to your ~/.bashrc
source "${SCRIPT_DIR}"/venv/bin/activate

pip install --upgrade pip setuptools wheel

# CUDA
pip install -r "${SCRIPT_DIR}"/requirements.txt
# CPU
#pip install -r "${SCRIPT_DIR}"/requirements_cpu.txt

spacy download en_core_web_sm
#spacy download en_core_web_trf
#python "${SCRIPT_DIR}"/setup.py install
