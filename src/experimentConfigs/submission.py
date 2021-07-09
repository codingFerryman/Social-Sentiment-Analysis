import random
import sys
from pathlib import Path

import pandas as pd
import torch
from tqdm import trange
from tqdm.auto import tqdm
from transformers import TextClassificationPipeline, AutoModelForSequenceClassification, AutoTokenizer

from utils import get_data_path
from utils.cleaningText import cleaning_default


class TransformersPredict:
    def __init__(self, load_path, text_path, cuda_device=None, is_test=True):
        self.is_test = is_test

        if cuda_device is None:
            if torch.cuda.is_available():
                cuda_device = -1
            else:
                cuda_device = 0

        model_path = Path(load_path, 'model')
        tokenizer_path = Path(load_path, 'tokenizer')
        model = AutoModelForSequenceClassification.from_pretrained(model_path)
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
        self.pipeline = TextClassificationPipeline(model=model, tokenizer=tokenizer, device=cuda_device,
                                                   binary_output=True)
        with open(text_path, 'r') as fp:
            lines = fp.readlines()
        self.data = self.pre_process_test(lines)

        self.text_path = text_path
        self.load_path = load_path
        self.pred = None

    def predict(self, batch_size=128):
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
        pred_df.to_csv(save_path, index=False)

    def pre_process_test(self, lines: list):
        # Cleaning test set
        if self.is_test:
            data = [s.split(',', 1)[-1] for s in lines]
            ids = [s.split(',', 1)[0] for s in lines]
        else:
            data = []
            ids = []
            for s in tqdm(lines):
                _tmp = s.split('\u0001')
                data.append(_tmp[-1])
                ids.append(_tmp[0])
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
    argv = {a.split('=')[0]: a.split('=')[1] for a in args[1:]}
    load_path = argv.get('load_path', None)
    text_path = argv.get('text_path', None)
    batch_size = argv.get('batch_size', 2000)
    cuda_device = argv.get('cuda', None)

    if load_path is None:
        print("No load_path specified")
        exit(0)

    if text_path is None:
        data_path = get_data_path()
        _text_path = Path(data_path, 'test_data.txt')
        if _text_path.is_file():
            text_path = _text_path
        else:
            print("No text_path specified")
            exit(0)

    trans_predict = TransformersPredict(load_path=load_path, text_path=text_path, cuda_device=cuda_device)
    trans_predict.predict(batch_size=batch_size)
    trans_predict.submission_file()


if __name__ == '__main__':
    main(sys.argv)
