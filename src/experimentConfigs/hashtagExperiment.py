import json
import time
from pathlib import Path
import sys
import os
import torch
from sklearn.metrics import accuracy_score
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import AutoConfig, AutoModelForSequenceClassification, AutoTokenizer
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from preprocessing import PretrainedTransformersPipeLine
from utils import get_project_path, loggers
from experimentConfigs.experiment import report

logger = loggers.getLogger('HashtagExperiment', debug=1)

PROJECT_DIRECTORY = get_project_path()
# timestamp = '20210702-222212'
# checkpoint = 'checkpoint-4268'
# model_name = 'roberta-base'

timestamp = '20210702-224602'
model_name = 'vinai/bertweet-base'
checkpoint = 'checkpoint-3201'

device = 'cuda:1'
data_loaded_ratio = 0.1
batch_size = 3000

freq_threshold = 10
prob_threshold = 0.7

last_state_path = Path(PROJECT_DIRECTORY, 'trainings', 'logging',
                       model_name, timestamp, 'checkpoints', checkpoint)

config = AutoConfig.from_pretrained(last_state_path)
model = AutoModelForSequenceClassification.from_config(config).to(device)
tokenizer = AutoTokenizer.from_pretrained(model_name)

pipeline = PretrainedTransformersPipeLine(model_name)
pipeline.loadData(ratio=data_loaded_ratio)

data, labels = pipeline.randomizeAllData()

# Evaluation
def eval_predict(data):
    data_loader = DataLoader(data, batch_size=batch_size)
    _preds = torch.tensor([], device=device)
    _logits = torch.tensor([], device=device)
    with torch.no_grad():
        for sent in tqdm(data_loader):
            inputs = tokenizer(sent, truncation=True, padding='max_length', max_length=64, return_tensors="pt")
            inputs = inputs.to(device)
            logit = model(**inputs).logits
            _logits = torch.cat((_logits, logit), 0)
            _pred = torch.argmax(torch.softmax(logit, dim=-1), dim=-1)
            _preds = torch.cat((_preds, _pred), 0)
    return _preds, _logits


with open('/home/he/Workspace/CIL/src/models/hashtag.json', 'r') as fp:
    hashtag_dict = json.load(fp)

pred, logit = eval_predict(data)
accuracy = accuracy_score(pred.cpu(), labels)


def hashtag_matters(data, logits):
    pred_prob = torch.softmax(logits, dim=-1)
    for idx, sent in enumerate(tqdm(data)):
        _words = sent.split()
        _neg_prob = 0
        _pos_prob = 0
        for _w in _words:
            if _w[0] == '#':
                if _w[1:] in hashtag_dict.keys():
                    tag = _w[1:]
                    neg_freq = hashtag_dict[tag]['NegFreq']
                    pos_freq = hashtag_dict[tag]['PosFreq']
                    neg_ratio = hashtag_dict[tag]['NegRatio']
                    pos_ratio = hashtag_dict[tag]['PosRatio']
                    if (pos_freq > freq_threshold or neg_freq > freq_threshold) and (neg_ratio > prob_threshold or pos_ratio > prob_threshold):
                        _neg_prob += pred_prob[idx][0] + neg_ratio
                        _pos_prob += pred_prob[idx][1] + pos_ratio
        if _neg_prob * _pos_prob != 0:
            pred_prob[idx] = torch.softmax(torch.tensor([_neg_prob, _pos_prob], device=device), dim=-1)
    pred = torch.argmax(pred_prob, dim=-1)
    return pred


hashtag_pred = hashtag_matters(data, logit)
hashtag_accuracy = accuracy_score(hashtag_pred.cpu(), labels)

result = {'experiment_time': time.strftime("%Y%m%d-%H%M%S"),
          'model': model_name,
          'model_timestamp': timestamp,
          'model_checkpoint': checkpoint,
          'data_loaded': str(data_loaded_ratio * 100) + '%',
          'original_accuracy': str(accuracy),
          'hashtag_accuracy_neg': str(hashtag_accuracy),
          'frequency_threshold': str(freq_threshold),
          'ratio_threshold': str(prob_threshold)
          }

logger.debug(result)

save_path = Path(PROJECT_DIRECTORY, 'trainings', 'hashtagExperimentResults.json')
report(result, save_path)
# prepend_multiple_lines(file_name=save_path, list_of_lines=[json.dumps(result, indent=4)])

# def preProcessTest(data_test: list):
#     # Cleaning test set
#     data_test = [s.split(',', 1)[-1] for s in data_test]
#     min_test, max_test, zero_len_idx_test = get_min_max(data_test)
#     ic(min_test, max_test, zero_len_idx_test, len(data_test))
#     test_text = list(filter(lambda x: x != "", data_test))
#     ic(len(test_text))
#     # The ids of the items in test_text
#     test_id = list(set(range(1, len(data_test) + 1)) - set(zero_len_idx_test))
#     return {
#         'text': test_text,
#         'ids': test_id,
#         'min_len': min_test,
#         'max_len': max_test,
#         'zero_len_ids': zero_len_idx_test
#     }
