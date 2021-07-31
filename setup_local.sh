#!/bin/bash

# This script is for setting up the environment on the local machine
# You should have python3.8-venv installed on your computer

if [ ! -d ./.git/ ]
then
git init
fi

CIL_LOCALREPO=$(cd "$(dirname "${BASH_SOURCE[0]}")" &>/dev/null && pwd)

chmod +x ${CIL_LOCALREPO}/setup_dataset.sh
chmod +x ${CIL_LOCALREPO}/setup_environment.sh

${CIL_LOCALREPO}/setup_dataset.sh

python -m venv "${CIL_LOCALREPO}"/venv
source "${CIL_LOCALREPO}"/venv/bin/activate
pip install --upgrade pip setuptools wheel

# CUDA
pip install -r "${CIL_LOCALREPO}"/requirements.txt
# CPU
#pip install -r "${SCRIPT_DIR}"/requirements_cpu.txt

spacy download en_core_web_sm
#python "${SCRIPT_DIR}"/setup.py install
