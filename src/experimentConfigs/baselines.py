import os
import random
import sys

import torch
from datasets import load_metric
from tqdm import trange
from transformers import TextClassificationPipeline, AutoModelForSequenceClassification, AutoTokenizer
from transformers import logging as hf_logging

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from utils.others import set_seed
from utils.inputFunctions import loadData
from utils.loggers import getLogger
from preprocessing.cleaningText import cleaning_default

set_seed()
logger = getLogger("Baseline", debug=True)
hf_logging.set_verbosity_error()
device = 0 if torch.cuda.is_available() else -1

# Load and preprocess the dataset
pos, neg, _ = loadData()

pos = cleaning_default(pos)
neg = cleaning_default(neg)

pos = list(filter(None, pos))
neg = list(filter(None, neg))

X = pos + neg
y = [1] * len(pos) + [-1] * len(neg)

all_data = list(zip(X, y))
random.shuffle(all_data)
X, y = zip(*all_data)
X = list(X)
y = list(y)
logger.debug("dataset loaded")


# Predict
def predict(data, pipeline, batch_size=128):
    _results = []
    tr = trange(0, len(data), batch_size)
    for i in tr:
        _results += pipeline(data[i:i + batch_size])
    return _results


# Accuracy
def compute_metrics(predictions, labels):
    predictions = [_pred['label'] for _pred in predictions]
    metric = load_metric("accuracy")
    result = metric.compute(predictions=predictions, references=labels)
    logger.debug("The metrics in this eval: {}".format(str(result)))
    return result


# Initialize pipelines for baselines then predict and evaluate
model_configs = {
    "num_labels": 2,
    "id2label": {
        1: 1,
        0: -1
    }
}

roberta_pipeline = TextClassificationPipeline(
    model=AutoModelForSequenceClassification.from_pretrained('roberta-base', **model_configs),
    tokenizer=AutoTokenizer.from_pretrained('roberta-base'),
    framework='pt',
    task='sentiment-analysis-roberta',
    binary_output=True,
    device=device
)
logger.debug("roberta_pipeline initialized")

roberta_result = predict(X, roberta_pipeline, batch_size=1024)
roberta_accuracy = compute_metrics(roberta_result, y)
logger.info(f"roberta_accuracy: {roberta_accuracy}")

bert_pipeline = TextClassificationPipeline(
    model=AutoModelForSequenceClassification.from_pretrained('bert-base-uncased', **model_configs),
    tokenizer=AutoTokenizer.from_pretrained('bert-base-uncased'),
    framework='pt',
    task='sentiment-analysis-bert',
    binary_output=True,
    device=device
)
logger.debug("bert_pipeline initialized")

bert_result = predict(X, bert_pipeline, batch_size=1024)
bert_accuracy = compute_metrics(bert_result, y)
logger.info(f"bert_accuracy: {bert_accuracy}")

xlnet_pipeline = TextClassificationPipeline(
    model=AutoModelForSequenceClassification.from_pretrained('xlnet-base-cased', **model_configs),
    tokenizer=AutoTokenizer.from_pretrained('xlnet-base-cased'),
    framework='pt',
    task='sentiment-analysis-xlnet',
    binary_output=True,
    device=device
)
logger.debug("xlnet_pipeline initialized")

xlnet_result = predict(X, xlnet_pipeline, batch_size=512)
xlnet_accuracy = compute_metrics(xlnet_result, y)
logger.info(f"xlnet_accuracy: {xlnet_accuracy}")

bertweet_pipeline = TextClassificationPipeline(
    model=AutoModelForSequenceClassification.from_pretrained('vinai/bertweet-base', **model_configs),
    tokenizer=AutoTokenizer.from_pretrained('vinai/bertweet-base'),
    framework='pt',
    task='sentiment-analysis-bertweet',
    binary_output=True,
    device=device
)
logger.debug("bertweet_pipeline initialized")

bertweet_result = predict(X, bertweet_pipeline, batch_size=1024)
bertweet_accuracy = compute_metrics(bertweet_result, y)
logger.info(f"bertweet_accuracy: {bertweet_accuracy}")
