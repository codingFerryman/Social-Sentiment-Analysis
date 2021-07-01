#!/bin/bash

# The next line should be executed directly in your terminal again
# ... if you want to run $python $PYTHON_FILE.py directly
module load gcc/6.3.0 python_gpu/3.8.5 hdf5/1.10.1

python -m venv ./cil_venv

# The next line should be executed directly in your terminal again
# ... if you want to run $python $PYTHON_FILE.py directly
source ./cil_venv/bin/activate

pip install --upgrade pip setuptools wheel
pip install torch==1.8.1+cu102 -f https://download.pytorch.org/whl/lts/1.8/torch_lts.html
pip install 'torchtext<0.10'
pip install --upgrade 'spacy[cuda102,transformers,lookups]'
spacy download en_core_web_sm
spacy download en_core_web_trf
pip install -r requirements.txt
