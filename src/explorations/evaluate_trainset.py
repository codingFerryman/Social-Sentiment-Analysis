import os
import random
import sys
from pathlib import Path
from typing import Union

import pandas as pd

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from utils import get_data_path, loggers
from experimentConfigs.submission import TransformersPredict

logger = loggers.getLogger("EvaluateTrainDataset", True)

data_path = get_data_path()


class TransformersPredictEval(TransformersPredict):
    def __init__(self, load_path: Union[str, Path],
                 fast_tokenizer: bool = False,
                 full_or_sub: str = None,
                 text_path: Union[str, Path] = None,
                 pos_path: Union[str, Path] = None,
                 neg_path: Union[str, Path] = None,
                 device: str = None, is_test: bool = False):
        # Generate the data file by combine and shuffle pos and neg data if no text_path
        if text_path is None:
            # Adapt inputs to real filenames
            assert full_or_sub in ['sub', 'full'], 'full_or_sub should be full or sub'
            if full_or_sub == 'sub':
                full_or_sub_file_suffix = ''
            elif full_or_sub == 'full':
                full_or_sub_file_suffix = '_full'
            self.full_or_sub = full_or_sub
            if pos_path is None:
                pos_path = Path(data_path, 'train_pos{0}.txt'.format(full_or_sub_file_suffix))
            if neg_path is None:
                neg_path = Path(data_path, 'train_neg{0}.txt'.format(full_or_sub_file_suffix))
            logger.debug('Loading data from:')
            logger.debug(pos_path)
            logger.debug(neg_path)

            # Read data files and set the labels
            tmp = []
            with open(pos_path, 'r') as fp:
                pos_text = list(set(fp.readlines()))
                pos_data = list(zip([1] * len(pos_text), pos_text))
            with open(neg_path, 'r') as fn:
                neg_text = list(set(fn.readlines()))
                neg_data = list(zip([-1] * len(neg_text), neg_text))
            tmp.extend(pos_data)
            tmp.extend(neg_data)
            random.shuffle(tmp)
            data_indexed = [(i,) + d for i, d in enumerate(tmp, 1)]
            # ... then write the data to a file. The separator here is invisible
            data2write = ['\u0001'.join(str(d) for d in doc) for doc in data_indexed]
            text_path = Path(data_path, self.full_or_sub + '_data.txt')
            with open(text_path, 'w') as ft:
                ft.writelines(data2write)
        else:
            if 'full' in Path(text_path).stem:
                self.full_or_sub = 'full'
            elif 'sub' in Path(text_path).stem:
                self.full_or_sub = 'sub'
            else:
                self.full_or_sub = 'new'

        # Init superclass
        super(TransformersPredictEval, self).__init__(load_path, text_path, fast_tokenizer, device, is_test)

    def evaluation_file(self, save_path=None):
        if save_path is None:
            save_path = Path(self.load_path, 'pred_train_' + self.full_or_sub + '.csv')
        logger.info('The evaluation file will be saved to ' + str(save_path))
        # The golden data read from the file
        data_indexed = pd.read_csv(self.text_path, sep='\u0001', names=['index', 'Golden', 'Text'])
        data_indexed['Text'] = data_indexed['Text'].str.strip()

        # The prediction labels, scores, and the index of zero length sentences
        pred_labels = self.get_predictions().tolist()
        pred_score = self.get_scores().tolist()
        id_zero_len = self.data['zero_len_ids']

        # Randomly predict the label of zero length sentences, and set their probability to 0.5
        pred_id = self.data['ids'] + id_zero_len
        pred_est = pred_labels + [random.choice([-1, 1]) for _ in range(len(id_zero_len))]
        pred_est_score = pred_score + [0.5 for _ in range(len(id_zero_len))]
        # ... and save the predictions to DataFrame
        pred_df_dict = {'index': pred_id,
                        'pred': pred_est,
                        'score': pred_est_score}
        pred_df = pd.DataFrame.from_dict(pred_df_dict)

        # Join the prediction DataFrame and the golden DataFrame
        results = pred_df.rename(columns={'index': 'Id', 'pred': 'Prediction', 'score': 'Score'})
        # ... and write it to a csv file
        results.to_csv(save_path, index=False)


def main(args: list):
    """
    The main function of the evaluation. It will predict the sentiment of texts in the train data
    Args:
        args (list): a dictionary containing the program arguments (sys.argv)
        - load_path: The root directory containing 'model' and 'tokenizer'
        - batch_size: The batch size in prediction
        - device: The device for prediction.
            If not given, the program will automatically use the first cuda device otherwise the cpu
        - fast_tokenizer: Use Fast Tokenizer or not in predictions. Better to use the same as training tokenizer
            Using fast tokenizer by default
        - text_path: The text file to be processed. It will overwrite the full_or_sub
        - full_or_sub: Use full or sub dataset. It will be ignored if text_path is given
    Call it like:
        python explorations/evaluate_trainset.py \
            load_path=trainings/MODEL_NAME/TIMESTAMP \
            fast_tokenizer=false full_or_sub=sub
    """
    argv = {a.split('=')[0]: a.split('=')[1] for a in args[1:]}

    load_path = argv.get('load_path', None)
    assert load_path, "No load_path specified"

    batch_size = argv.get('batch_size', 256)

    device = argv.get('device', None)

    fast_tokenizer = argv.get('fast_tokenizer', 'true').lower()
    assert fast_tokenizer in ['true', 'false']
    fast_tokenizer = False if 'f' in fast_tokenizer else True

    text_path = argv.get('text_path', None)
    full_or_sub = argv.get('full_or_sub', None)
    assert text_path or full_or_sub, 'text_path or full_or_sub should be given'

    if text_path is None:
        assert full_or_sub in ['full', 'sub'], 'full_or_sub should be full or sub'
        if full_or_sub == 'full':
            _text_path = Path(data_path, 'full_data.txt')
        elif full_or_sub == 'sub':
            _text_path = Path(data_path, 'sub_data.txt')
        if _text_path.is_file():
            logger.info("Using the data file which already exists.")
            text_path = _text_path

    trans_predict = TransformersPredictEval(load_path=load_path, text_path=text_path, device=device,
                                            fast_tokenizer=fast_tokenizer, full_or_sub=full_or_sub)
    trans_predict.predict(batch_size=batch_size)
    trans_predict.evaluation_file()


if __name__ == '__main__':
    main(sys.argv)
