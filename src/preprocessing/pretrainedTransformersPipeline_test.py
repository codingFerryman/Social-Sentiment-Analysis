import unittest
import numpy as np
import sklearn
from sklearn import model_selection
import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
import pretrainedTransformersPipeline
import loggers
import inputFunctions

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

    # def test_loading(self):
    #     newBowp = pretrainedTransformersPipeline.PretrainedTransformersPipeLine(loadFunction=inputFunctions.loadDataForUnitTesting)
    #     newBowp.loadData()

    # def test_train(self):
    #     newBowp = pretrainedTransformersPipeline.PretrainedTransformersPipeLine(loadFunction=inputFunctions.loadDataForUnitTesting)
    #     newBowp.loadData()
    #     newBowp.trainTokenizer()
    #     print(newBowp.num_words)        

    # def test_textsToPaddedSequences(self):
    #     newBowp = pretrainedTransformersPipeline.PretrainedTransformersPipeLine(loadFunction=inputFunctions.loadDataForUnitTesting)   
    #     newBowp.loadData()
    #     newBowp.trainTokenizer()
    #     posTextPaddedSequences = newBowp.textsPosToPaddedSequences()
    #     negTextPaddedSequences = newBowp.textsNegToPaddedSequences()
    #     print(posTextPaddedSequences)
    #     print(negTextPaddedSequences)

    
    def test_argmixPositiveNegative(self):
        logger.debug("Testing argmixPositiveNegative")
        newBowp = pretrainedTransformersPipeline.PretrainedTransformersPipeLine(loadFunction=inputFunctions.loadDataForUnitTesting)  
        newBowp.loadData()
        newBowp.trainTokenizer()
        posTextPaddedSequences = newBowp.textsPosToPaddedSequences()
        negTextPaddedSequences = newBowp.textsNegToPaddedSequences()
        argMix = newBowp.argmixPositiveNegative(posTextPaddedSequences, negTextPaddedSequences)
        # print(len(posTextPaddedSequences), len(negTextPaddedSequences))
        sumLen = len(posTextPaddedSequences) + len(negTextPaddedSequences)
        self.assertEqual(sumLen, len(argMix))
        countZeros = 0
        countOnes = 0
        for a in argMix:
            if a == 0:
                countZeros = countZeros + 1
            else:
                countOnes = countOnes + 1
        # print(countZeros, countOnes)
        self.assertEqual(countZeros, len(negTextPaddedSequences))
        self.assertEqual(countOnes, len(posTextPaddedSequences))
    

    def test_getEncodedDataset(self):
        logger.debug("Testing getEncodedDataset")
        newBowp = pretrainedTransformersPipeline.PretrainedTransformersPipeLine(loadFunction=inputFunctions.loadDataForUnitTesting)  
        newBowp.loadData()
        encDataTrain, encDataVal = newBowp.getEncodedDataset(splitter=lambda *x: model_selection.train_test_split(*x, test_size=0.4))
        # print(encDataTrain, encDataVal)

if __name__ == "__main__":
    unittest.main()