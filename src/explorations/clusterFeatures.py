# import 1
import os
import sys

sys.path.append(os.path.join(os.getcwd(), '..'))

# import 2
from experimentConfigs.submission import TransformersPredict
import numpy as np

load_path ="/home/sniper/projects_local/CIL/Computational-Intelligence-Lab/trainings/roberta-base/20210709-102233"
text_path_pos ="/home/sniper/projects_local/CIL/Computational-Intelligence-Lab/data/train_pos.txt"
text_path_neg ="/home/sniper/projects_local/CIL/Computational-Intelligence-Lab/data/train_neg.txt"
batch_size=32

try:
    trans_predictPos = TransformersPredict(load_path=load_path, text_path=text_path_pos)
    for h in trans_predictPos.extractHiddenStates(batch_size=batch_size, appendToList=True):
        pass
except KeyboardInterrupt:
    print("Keyboard interupt, progress till now will be saved")
    pass
vecReprPos = trans_predictPos.getVectorRepresentation()
with open('test.npy', 'wb') as fw:
    np.save(fw, vecReprPos)
