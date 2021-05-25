import abc
import typing
import numpy as np
ModelConstruction = object
def get_iterator_splitter_from_name(it_name: str):
    return {
        "train_test_split": [sklearn.model_selection.train_test_split]
    }[it_name]