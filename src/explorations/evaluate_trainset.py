import os
import random
import sys
from pathlib import Path

import pandas as pd

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from utils import get_data_path
from experimentConfigs.submission import TransformersPredict

data_path = get_data_path()

class TransformersPredictEval(TransformersPredict):
    def __init__(self, load_path, pos_path=None, neg_path=None, cuda_device=0, is_test=False):
        if pos_path is None:
            pos_path = Path(data_path, 'train_pos_full.txt')
        if neg_path is None:
            neg_path = Path(data_path, 'train_neg_full.txt')
        tmp = []
        with open(pos_path, 'r') as fp:
            pos_text = list(set(fp.readlines()))
            pos_data = list(zip([1]*len(pos_text), pos_text))
        with open(neg_path, 'r') as fn:
            neg_text = list(set(fn.readlines()))
            neg_data = list(zip([-1]*len(neg_text), neg_text))
        tmp.extend(pos_data)
        tmp.extend(neg_data)
        random.shuffle(tmp)
        self.data_indexed = [(i,)+d for i, d in enumerate(tmp)]
        data2write = ['\u0001'.join(str(d) for d in doc) for doc in self.data_indexed]
        text_path = Path(data_path, 'full_data.txt')
        with open(text_path, 'w') as ft:
            ft.writelines(data2write)
        super(TransformersPredictEval, self).__init__(load_path, text_path, cuda_device, is_test)

    def evaluation_file(self, save_path='./evaluation_on_train.csv'):
        pred_labels = [r['label'] for r in self.pred]
        pred_score = [r['score'] for r in self.pred]
        id_zero_len = self.data['zero_len_ids']
        pred_id = self.data['ids'] + id_zero_len
        pred_est = pred_labels + [random.choice([-1, 1]) for _ in range(len(id_zero_len))]
        pred_est_score = pred_score + [0.5 for _ in range(len(id_zero_len))]
        golden_data = [self.data_indexed[int(_idx)] for _idx in pred_id]
        golden_label = [idg[1] for idg in golden_data]
        original_text = [idg[2].rstrip() for idg in golden_data]
        pred_dict = {'Id': pred_id,
                     'Golden': golden_label,
                     'Prediction': pred_est,
                     'Score': pred_est_score,
                     'Text': original_text}
        pred_df = pd.DataFrame(pred_dict)
        pred_df.to_csv(save_path, index=False)


def main(args: list):
    argv = {a.split('=')[0]: a.split('=')[1] for a in args[1:]}
    load_path = argv.get('load_path', None)
    batch_size = argv.get('batch_size', 256)
    if load_path is None:
        print("No load_path specified")
        exit(0)

    tpe = TransformersPredictEval(load_path=load_path)
    tpe.predict(batch_size=batch_size)
    tpe.evaluation_file()


if __name__ == '__main__':
    main(sys.argv)
