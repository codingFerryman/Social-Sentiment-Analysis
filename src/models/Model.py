import sklearn
from sklearn.model_selection import cross_validate, StratifiedKFold, train_test_split

ModelConstruction = object


# class TrainTestSplit:
#     def split(self, X: list, y=None, test_size: float = 0.1) -> typing.Tuple[np.ndarray, np.ndarray]:
#         xLen = len(X)
#         train_size = int(xLen * (1 - test_size))
#         train_index = np.random.choice(range(xLen), size=train_size)
#         test_index = np.array([i for i in range(xLen) if not i in train_index])
#         yield (train_index, test_index)

StratifiedKFoldInstance = StratifiedKFold(n_splits=5)

def stratifiedKFoldSplit(X, y, **kwargs):
    for train_index, test_index in StratifiedKFoldInstance.split(X, y):
        X_train, X_test = [X[i] for i in train_index], [X[i] for i in test_index]
        y_train, y_test = [y[i] for i in train_index], [y[i] for i in test_index]
        yield  X_train, X_test, y_train, y_test

def trainTestSplit(X, y, **kwargs):
    yield train_test_split(X,y,**kwargs)

def get_iterator_splitter_from_name(it_name: str):
    return {
        "train_test_split": trainTestSplit,
        # "cross_val_score": sklearn.model_selection.cross_val_score,  # it can be performed via stratifiedKfold
        # "cross_validate_accuracy": lambda clf, X, y: cross_validate(clf, X, y, scoring=['accuracy']), # it is performed via stratifiedKfold
        "stratifiedKfold": stratifiedKFoldSplit
    }[it_name]
