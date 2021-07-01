import json
import time
from pathlib import Path

import torch
from icecream import ic
from sklearn.metrics import accuracy_score
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import AutoConfig, AutoModelForSequenceClassification, AutoTokenizer

from preprocessing import PretrainedTransformersPipeLine
from utils import get_project_path, prepend_multiple_lines

PROJECT_DIRECTORY = get_project_path()
# timestamp = '20210629-013136'
# checkpoint = 'checkpoint-5335'
# model_name = 'cardiffnlp/twitter-roberta-base-sentiment'

timestamp = '20210629-140517'
checkpoint = 'checkpoint-801'
model_name = 'roberta-base'

last_state_path = Path(PROJECT_DIRECTORY, 'trainings', 'logging',
                       model_name, timestamp, 'checkpoints', checkpoint)
device = 'cuda:1'
data_loaded_ratio = 0.5

config = AutoConfig.from_pretrained(last_state_path)
model = AutoModelForSequenceClassification.from_config(config).to(device)
tokenizer = AutoTokenizer.from_pretrained(model_name)

pipeline = PretrainedTransformersPipeLine(model_name)
pipeline.loadData(ratio=data_loaded_ratio)

pos_data = pipeline.dataPos
neg_data = pipeline.dataNeg


# Evaluation
def eval_predict(data):
    data_loader = DataLoader(data, batch_size=2500)
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


pos_pred, pos_logit = eval_predict(pos_data)
accuracy_pos = accuracy_score(pos_pred.cpu(), [1] * len(pos_pred))

neg_pred, neg_logit = eval_predict(neg_data)
accuracy_neg = accuracy_score(neg_pred.cpu(), [0] * len(neg_pred))

ic(accuracy_pos, accuracy_neg)

with open('/home/he/Workspace/CIL/src/models/hashtag.json', 'r') as fp:
    hashtag_dict = json.load(fp)


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
                    if pos_freq > 10 or neg_freq > 10:
                        _neg_prob += pred_prob[idx][0] + hashtag_dict[tag]['NegRatio']
                        _pos_prob += pred_prob[idx][1] + hashtag_dict[tag]['PosRatio']
        if _neg_prob * _pos_prob != 0:
            pred_prob[idx] = torch.softmax(torch.tensor([_neg_prob, _pos_prob], device=device), dim=-1)
    pred = torch.argmax(pred_prob, dim=-1)
    return pred


hashtag_pos_pred = hashtag_matters(pos_data, pos_logit)
hashtag_accuracy_pos = accuracy_score(hashtag_pos_pred.cpu(), [1] * len(hashtag_pos_pred))

hashtag_neg_pred = hashtag_matters(neg_data, neg_logit)
hashtag_accuracy_neg = accuracy_score(hashtag_neg_pred.cpu(), [0] * len(hashtag_neg_pred))

ic(hashtag_accuracy_pos, hashtag_accuracy_neg)

result = {'experiment_time': time.strftime("%Y%m%d-%H%M%S"),
          'model': model_name,
          'model_timestamp': timestamp,
          'model_checkpoint': checkpoint,
          'data_loaded': str(data_loaded_ratio * 100) + '%',
          'original_pos_accuracy': str(accuracy_pos),
          'original_neg_accuracy': str(accuracy_neg),
          'hashtag_accuracy_pos': str(hashtag_accuracy_pos),
          'hashtag_accuracy_neg': str(hashtag_accuracy_neg),
          }

save_path = Path(PROJECT_DIRECTORY, 'trainings', 'hashtagExperimentResults.json')
prepend_multiple_lines(file_name=save_path, list_of_lines=[json.dumps(result, indent=4)])

# def pre_process_test(data_test: list):
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
