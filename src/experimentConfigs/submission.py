from pathlib import Path

from transformers import TextClassificationPipeline, AutoModelForSequenceClassification, AutoTokenizer
from preprocessing.pretrainedTransformersPipeline import TwitterDatasetTorch
from utils import get_project_path


class TransformersPredict:
    def __init__(self, model_path, model_name, text_path, cuda_device=-1):
        model = AutoModelForSequenceClassification.from_pretrained(model_path)
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.pipeline = TextClassificationPipeline(model=model, tokenizer=tokenizer, device=cuda_device)
        self.text_path = text_path

    def predict(self):
        with open(self.text_path, 'r') as fp:
            lines = fp.readlines()
        data = self.pre_process_test(lines)
        _pred = self.pipeline(data['text'])
        return _pred

    @staticmethod
    def pre_process_test(data: list):
        # Cleaning test set
        data = [s.split(',', 1)[-1] for s in data][:50]
        data = list(map(TwitterDatasetTorch.cleaning, data))
        text = list(filter(lambda x: x != "", data))
        zero_len_idx = []
        for idx, t in enumerate(text):
            t_len = len(t.split())
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

    timestamp = '20210704-164012'
    model_name = 'vinai/bertweet-base'
    checkpoint = 'checkpoint-15627'

    model_path = Path(PROJECT_DIRECTORY, 'trainings', 'logging',
                      model_name, timestamp, 'checkpoints', checkpoint)

    predict_pipeline = TransformersPredict(model_path, model_name, test_path, cuda_device=0)
    pred = predict_pipeline.predict()
