import unittest

import numpy as np
from sklearn import model_selection

import inputFunctions
import loggers
# sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
import pretrainedTransformersPipeline

logger = loggers.getLogger("PretrainedTransformersPipeLineTest", debug=True)


class PretrainedTransformersPipeLineTest(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super(PretrainedTransformersPipeLineTest, self).__init__(*args, **kwargs)

    def assertArrayEqual(self, arr1: np.ndarray, arr2: np.ndarray):
        for d1, d2 in zip(arr1.shape, arr2.shape):
            self.assertEqual(d1, d2)
        for i in range(arr1.size):
            # print(arr1.take(i))
            self.assertEqual(arr1.take(i), arr2.take(i))

    def test_loading(self):
        newBowp = pretrainedTransformersPipeline.PretrainedTransformersPipeLine(
            loadFunction=inputFunctions.loadDataForUnitTesting)
        newBowp.loadData()

    def test_getEncodedDataset(self):
        logger.debug("Testing getEncodedDataset")
        newBowp = pretrainedTransformersPipeline.PretrainedTransformersPipeLine(
            loadFunction=inputFunctions.loadDataForUnitTesting)
        newBowp.loadData()
        encDataTrain, encDataVal = newBowp.getEncodedDataset(
            splitter=lambda *x: model_selection.train_test_split(*x, test_size=0.4))


if __name__ == "__main__":
    unittest.main()
