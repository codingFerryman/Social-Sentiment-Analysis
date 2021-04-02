import os
import numpy as np

DATA_DIRECTORY = os.path.join(os.path.dirname(__file__), '..', 'data')

def loadData(dataDirectory:str=None):
    if dataDirectory== None:
        dataDirectory = DATA_DIRECTORY

    with open(os.path.join(dataDirectory, 'train_pos_full.txt'), 'r') as fp:
        train_pos_full = fp.readlines()

    with open(os.path.join(dataDirectory, 'train_neg_full.txt'), 'r') as fp:
        train_neg_full = fp.readlines()
    
    with open(os.path.join(dataDirectory, 'test_data.txt'), 'r') as fp:
        test = fp.readlines()

    return train_pos_full, train_neg_full, test


def loadDataForTesting(dataDirectory:str=None) -> list:
    if dataDirectory== None:
        dataDirectory = DATA_DIRECTORY
    
    with open(os.path.join(dataDirectory, 'test_data.txt'), 'r') as fp:
        test_data = fp.readlines()

    return test_data

def randomizeData(train_pos:list, train_neg:list):
    res_pos = np.random.shuffle(train_pos)
    res_neg = np.random.shuffle(train_neg)
    return res_pos, res_neg

# def addLabelsToData():
    