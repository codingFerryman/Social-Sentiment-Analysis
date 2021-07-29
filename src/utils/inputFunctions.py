import random
from pathlib import PurePath
from typing import Union, Tuple

import numpy as np
import pandas as pd

from .loggers import getLogger
from .others import get_data_path
from .others import set_seed

DATA_DIRECTORY = get_data_path()
logger = getLogger("InputPipeline", debug=True)
set_seed()

def loadData(dataDirectory: str = None, ratio: Union[str, float, int] = "full") -> Tuple[list, list, list]:
    """
    Load datasets
    Args:
        dataDirectory (str or pathlib.Path):
        ratio: "sub", "full"; Or a float number between 0 and 1, which is the proportion of the full set

    Returns:
        3 lists: Positive training set, Negative training set and Test set
    """
    if dataDirectory is None:
        dataDirectory = DATA_DIRECTORY

    with open(PurePath(dataDirectory, 'train_pos_full.txt'), 'r', encoding='utf-8') as fp:
        train_pos_full = fp.readlines()
    with open(PurePath(dataDirectory, 'train_neg_full.txt'), 'r', encoding='utf-8') as fp:
        train_neg_full = fp.readlines()
    with open(PurePath(dataDirectory, 'test_data.txt'), 'r', encoding='utf-8') as fp:
        test_full = fp.readlines()

    if type(ratio) is int:
        ratio = float(ratio)
    assert isinstance(ratio, str) or isinstance(ratio, float)
    if type(ratio) is float:
        if ratio <= 0 or ratio > 1:
            raise AttributeError(
                'The input should be \'full\', \'sub\', \'clean\', or a (float) number between 0 and 1')
        pos_num_samples = int(ratio * len(train_pos_full))
        neg_num_samples = int(ratio * len(train_neg_full))
        pos = random.sample(train_pos_full, pos_num_samples)
        neg = random.sample(train_neg_full, neg_num_samples)
    else:
        if ratio == 'full':
            pos = train_pos_full
            neg = train_neg_full
        elif ratio == 'sub':
            with open(PurePath(dataDirectory, 'train_pos.txt'), 'r', encoding='utf-8') as fp:
                train_pos_sub = fp.readlines()
            with open(PurePath(dataDirectory, 'train_neg.txt'), 'r', encoding='utf-8') as fp:
                train_neg_sub = fp.readlines()
            pos = train_pos_sub
            neg = train_neg_sub
        elif ratio == 'clean':
            with open(PurePath(dataDirectory, 'train_pos_full_clean.txt'), 'r', encoding='utf-8') as fp:
                pos = fp.readlines()
            with open(PurePath(dataDirectory, 'train_neg_full_clean.txt'), 'r', encoding='utf-8') as fp:
                neg = fp.readlines()
            with open(PurePath(dataDirectory, 'test_data_clean.txt'), 'r', encoding='utf-8') as fp:
                test_full = fp.readlines()
        elif ratio == 'aug':
            with open(PurePath(dataDirectory, 'train_pos_full_aug.txt'), 'r', encoding='utf-8') as fp:
                pos = fp.readlines()
            with open(PurePath(dataDirectory, 'train_neg_full_aug.txt'), 'r', encoding='utf-8') as fp:
                neg = fp.readlines()
            with open(PurePath(dataDirectory, 'test_data.txt'), 'r', encoding='utf-8') as fp:
                test_full = fp.readlines()
        else:
            raise AttributeError(
                'The input should be \'full\', \'sub\', \'clean\', or a (float) number between 0 and 1')
    logger.debug(f"Before preprocessing: Positive: {len(pos)}, Negative: {len(neg)}, Test: {len(test_full)}")
    pos, neg = preprocessing(pos, neg)
    logger.info(f"Dataset loaded!")
    logger.info(f"Number of sentences: Positive: {len(pos)}, Negative: {len(neg)}, Test: {len(test_full)}")
    return pos, neg, test_full


def preprocessing(pos: list, neg: list) -> Tuple[list, list]:
    to_be_removed = r'(<.*?>)|(\.{3})'

    pos_df = pd.DataFrame({'text': pos, 'label': [1] * len(pos)})
    pos_df['text'] = pos_df['text'].str.strip().replace(to_be_removed, '')
    pos_count = pos_df.groupby(['text']).count()

    neg_df = pd.DataFrame({'text': neg, 'label': [-1] * len(neg)})
    neg_df['text'] = neg_df['text'].str.strip().replace(to_be_removed, '')
    neg_count = neg_df.groupby(['text']).count()

    inter_df = pos_count.join(neg_count, on='text', how='inner', lsuffix='_pos', rsuffix='_neg').reset_index()

    to_pos = np.select(inter_df['label_pos'] > inter_df['label_neg'], inter_df['text']).tolist()
    to_neg = np.select(inter_df['label_pos'] < inter_df['label_neg'], inter_df['text']).tolist()

    pos_result = list(filter(None, set(pos) - set(to_neg)))
    neg_result = list(filter(None, set(neg) - set(to_pos)))

    return pos_result, neg_result


def loadDataForUnitTesting(dataDirectory: str = None, ratio: float = 0.0002) -> Tuple[list, list, list]:
    return loadData(dataDirectory, ratio)


def randomizeData(train_pos: list, train_neg: list) -> Tuple[np.ndarray, np.ndarray]:
    res_pos = np.random.shuffle(train_pos)
    res_neg = np.random.shuffle(train_neg)
    return res_pos, res_neg
