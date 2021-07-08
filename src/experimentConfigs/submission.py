import random
from pathlib import Path

import pandas as pd
from tqdm import trange
from tqdm.auto import tqdm
from transformers import TextClassificationPipeline, AutoModelForSequenceClassification, AutoTokenizer

from preprocessing.pretrainedTransformersPipeline import TwitterDatasetTorch
from utils import get_project_path


class TransformersPredict:
    def __init__(self, load_path, text_path, cuda_device=-1, is_test=True):
        self.is_test = is_test

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
        self.pred = None

    def predict(self, batch_size=128):
        _results = []
        tr = trange(0, len(self.data['text']), batch_size)
        for i in tr:
            _results += self.pipeline(self.data['text'][i:i + batch_size])
        self.pred = _results
        return _results

    def submission_file(self, save_path='./submission.csv'):
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
            sent_proc = TwitterDatasetTorch.cleaning(sent)
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


if __name__ == '__main__':
    PROJECT_DIRECTORY = get_project_path()
    test_path = Path(PROJECT_DIRECTORY, 'data', 'test_data.txt')

    timestamp = '20210706-170133'
    model_name = 'roberta-base'

    load_path = Path(PROJECT_DIRECTORY, 'trainings',
                     model_name, timestamp)

    predict_pipeline = TransformersPredict(load_path, test_path, cuda_device=0)
    pred = predict_pipeline.predict(batch_size=2500)
    predict_pipeline.submission_file(Path(load_path, 'submission.csv'))
