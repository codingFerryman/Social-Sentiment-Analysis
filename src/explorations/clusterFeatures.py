#%%
import os
import sys
import numpy as np

sys.path.append(os.path.join(os.getcwd(), '..'))

# local imports
from experimentConfigs.submission import TransformersPredict
from utils.diskArray import DiskArray

#%% setting up paths
CIL_REPO = os.path.join(os.path.dirname(__file__), '..', '..')
TRAININGS_DIR = os.path.join(CIL_REPO, "trainings")
load_path = os.path.join(TRAININGS_DIR, "roberta-base/20210709-102233")
text_path_pos = os.path.join(CIL_REPO, "data/train_pos.txt")
text_path_neg = os.path.join(CIL_REPO, "data/train_neg.txt")
batch_size=32

#%% positive label features
daPos = DiskArray()
try:
    trans_predictPos = TransformersPredict(load_path=load_path, text_path=text_path_pos)
    for h in trans_predictPos.extractHiddenStates(batch_size=batch_size, appendToList=False):
        daPos.append(h)
except KeyboardInterrupt:
    print("Keyboard interupt, progress till now will be saved")
    pass
daPos.save("roberta-base-pos-features.diskArray")
#%% negative label features
daNeg = DiskArray()
try:
    trans_predictPos = TransformersPredict(load_path=load_path, text_path=text_path_neg)
    for h in trans_predictPos.extractHiddenStates(batch_size=batch_size, appendToList=False):
        daNeg.append(h)
except KeyboardInterrupt:
    print("Keyboard interupt, progress till now will be saved")
    pass
daNeg.save("roberta-base-neg-features.diskArray")