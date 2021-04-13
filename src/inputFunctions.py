import os
import numpy as np
import typing

DATA_DIRECTORY = os.path.join(os.path.dirname(__file__), '..', 'data')

def loadData(dataDirectory:str=None) -> typing.Tuple[list,list,list]:
    if dataDirectory== None:
        dataDirectory = DATA_DIRECTORY

    with open(os.path.join(dataDirectory, 'train_pos_full.txt'), 'r') as fp:
        train_pos_full = fp.readlines()

    with open(os.path.join(dataDirectory, 'train_neg_full.txt'), 'r') as fp:
        train_neg_full = fp.readlines()

    with open(os.path.join(dataDirectory, 'test_data.txt'), 'r') as fp:
        test = fp.readlines()

    return train_pos_full, train_neg_full, test


def loadDataForUnitTesting(dataDirectory:str=None) -> typing.Tuple[list,list,list]:
    if dataDirectory== None:
        dataDirectory = DATA_DIRECTORY
    with open(os.path.join(dataDirectory, 'train_pos_full.txt'), 'r') as fp:
        train_pos_full = fp.readlines()
        train_pos_full = train_pos_full[:256] # for faster testing
    with open(os.path.join(dataDirectory, 'train_neg_full.txt'), 'r') as fp:
        train_neg_full = fp.readlines()
        train_neg_full = train_neg_full[:256] # for faster testing
    with open(os.path.join(dataDirectory, 'test_data.txt'), 'r') as fp:
        test = fp.readlines()

    return train_pos_full, train_neg_full, test

def randomizeData(train_pos:list, train_neg:list) -> typing.Tuple[np.ndarray, np.ndarray]:
    res_pos = np.random.shuffle(train_pos)
    res_neg = np.random.shuffle(train_neg)
    return res_pos, res_neg

# def addLabelsToData():
    