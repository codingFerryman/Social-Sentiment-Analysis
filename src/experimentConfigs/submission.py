import os
import random
import sys
from pathlib import Path

import pandas as pd
import torch
from tqdm import trange
from tqdm.auto import tqdm
from transformers import TextClassificationPipeline, AutoModelForSequenceClassification, AutoTokenizer

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from utils import loggers
from utils.utils import get_data_path
from preprocessing.cleaningText import cleaning_default

logger = loggers.getLogger("PredictForSubmission", True)


class TransformersPredict:
    def __init__(self, load_path, text_path, fast_tokenizer, cuda_device=None, is_test=True, ):
        self.is_test = is_test

        if cuda_device is None:
            if torch.cuda.is_available():
                cuda_device = -1
            else:
                cuda_device = 0
        else:
            cuda_device = int(cuda_device)

        model_path = Path(load_path, 'model')
        tokenizer_path = Path(load_path, 'tokenizer')
        logger.info(f"Loading model from {load_path}")
        model = AutoModelForSequenceClassification.from_pretrained(model_path)
        try:
            tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, use_fast=fast_tokenizer)
        except Exception as e:
            # TODO: More reasonable exception (However OSError doesn't work)
            raise (e, "Please switch the value of fast tokenizer and try again")

        self.pipeline = TextClassificationPipeline(model=model, tokenizer=tokenizer, device=cuda_device,
                                                   binary_output=True)
        with open(text_path, 'r') as fp:
            lines = fp.readlines()
        self.data = self.pre_process_test(lines)

        self.text_path = text_path
        self.load_path = load_path
        self.pred = None

    def predict(self, batch_size=128):
        batch_size = int(batch_size)
        _results = []
        tr = trange(0, len(self.data['text']), batch_size)
        for i in tr:
            _results += self.pipeline(self.data['text'][i:i + batch_size])
        self.pred = _results
        return _results

    def submission_file(self, save_path=None):
        if save_path is None:
            save_path = Path(self.load_path, 'submission.csv')
        pred_labels = [r['label'] for r in self.pred]
        id_zero_len = self.data['zero_len_ids']
        pred_id = self.data['ids'] + id_zero_len
        pred_est = pred_labels + [random.choice([-1, 1]) for _ in range(len(id_zero_len))]
        pred_dict = {'Id': pred_id, 'Prediction': pred_est}
        pred_df = pd.DataFrame(pred_dict)
        pred_df = pred_df.astype({
            'Id': int,
            'Prediction': int
        })
        pred_df.to_csv(save_path, index=False)

    def pre_process_test(self, lines: list):
        if self.is_test:
            # Clean test set
            data = [s.split(',', 1)[-1] for s in lines]
            ids = [s.split(',', 1)[0] for s in lines]

        else:
            # Clean the train set
            data = []
            ids = []
            for s in tqdm(lines):
                _tmp = s.split('\u0001')
                data.append(_tmp[-1])
                ids.append(int(_tmp[0]))
        text_id = []
        text = []
        zero_len_idx = []
        for idx, sent in tqdm(zip(ids, data)):
            sent_proc = cleaning_default(sent)
            if len(sent_proc) != 0:
                text_id.append(idx)
                text.append(sent_proc)
            else:
                zero_len_idx.append(idx)
        return {
            'text': text,
            'ids': text_id,
            'zero_len_ids': zero_len_idx
        }


def main(args: list):
    """
    The main function of submission. It will predict the sentiment of texts in the text data
    Args:
        args (list): a dictionary containing the program arguments (sys.argv)
        - load_path: The root directory containing 'model' and 'tokenizer'
        - batch_size: The batch size in prediction
        - cuda_device: The index of cuda device for prediction.
            If not given, the program will automatically use the first cuda device otherwise the cpu
        - fast_tokenizer: Use Fast Tokenizer or not in predictions. Better to use the same as training tokenizer
            Using normal tokenizer by default
        - text_path: The text file to be processed. data/test_data.txt by default

    Returns:

    """
    argv = {a.split('=')[0]: a.split('=')[1] for a in args[1:]}
    load_path = argv.get('load_path', None)
    assert load_path, "No load_path specified"
    batch_size = argv.get('batch_size', 2000)
    cuda_device = argv.get('cuda', None)

    text_path = argv.get('text_path', None)
    fast_tokenizer = argv.get('fast_tokenizer', 'false').lower()
    assert fast_tokenizer in ['true', 'false']
    fast_tokenizer = False if 'f' in fast_tokenizer else True

    if text_path is None:
        data_path = get_data_path()
        _text_path = Path(data_path, 'test_data.txt')
        if _text_path.is_file():
            text_path = _text_path
        else:
            print("No text_path specified")
            exit(0)

    trans_predict = TransformersPredict(load_path=load_path, text_path=text_path, cuda_device=cuda_device,
                                        fast_tokenizer=fast_tokenizer)
    trans_predict.predict(batch_size=batch_size)
    trans_predict.submission_file()


if __name__ == '__main__':
    main(sys.argv)
