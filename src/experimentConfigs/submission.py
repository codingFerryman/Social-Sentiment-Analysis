import random
from pathlib import Path

import pandas as pd
from tqdm import trange
from transformers import TextClassificationPipeline, AutoModelForSequenceClassification, AutoTokenizer

from preprocessing.pretrainedTransformersPipeline import TwitterDatasetTorch
from utils import get_project_path


class TransformersPredict:
    def __init__(self, model_path, model_name, text_path, cuda_device=-1):
        model = AutoModelForSequenceClassification.from_pretrained(model_path)
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.pipeline = TextClassificationPipeline(model=model, tokenizer=tokenizer, device=cuda_device,
                                                   binary_output=True)
        self.text_path = text_path
        self.data = None
        self.pred = None

    def predict(self, batch_size=128):
        with open(self.text_path, 'r') as fp:
            lines = fp.readlines()
        self.data = self.pre_process_test(lines)
        _results = []
        tr = trange(0, len(self.data['text']), batch_size)
        for i in tr:
            _results += self.pipeline(self.data['text'][i:i + batch_size])
        self.pred = _results
        return _results

    def submission_file(self, save_path='./submission'):
        pred_labels = [r['label'] for r in self.pred]
        id_zero_len = self.data['zero_len_ids']
        pred_id = self.data['ids'] + id_zero_len
        pred_est = pred_labels + [random.choice([-1, 1]) for _ in range(len(id_zero_len))]
        pred_dict = {'Id': pred_id, 'Prediction': pred_est}
        pred_df = pd.DataFrame(pred_dict)
        pred_df.to_csv(save_path, index=False)

    @staticmethod
    def pre_process_test(data: list):
        # Cleaning test set
        data = [s.split(',', 1)[-1] for s in data]
        data = list(map(TwitterDatasetTorch.cleaning, data))
        text = list(filter(lambda x: x != "", data))
        zero_len_idx = []
        for idx, t in enumerate(data):
            t_len = len(t)
            if t_len == 0:
                zero_len_idx.append(idx)
        # The ids of the items in text
        test_id = list(set(range(1, len(data) + 1)) - set(zero_len_idx))
        return {
            'text': text,
            'ids': test_id,
            'zero_len_ids': zero_len_idx
        }


if __name__ == '__main__':
    PROJECT_DIRECTORY = get_project_path()
    test_path = '/home/he/Workspace/cil-project/data/test_data.txt'

    timestamp = '20210704-234431'
    model_name = 'siebert/sentiment-roberta-large-english'
    checkpoint = 'checkpoint-62500'

    model_path = Path(PROJECT_DIRECTORY, 'trainings', 'logging',
                      model_name, timestamp, 'checkpoints', checkpoint)

    predict_pipeline = TransformersPredict(model_path, model_name, test_path, cuda_device=0)
    pred = predict_pipeline.predict(batch_size=2500)
    predict_pipeline.submission_file(Path(model_path, 'submission.csv'))
