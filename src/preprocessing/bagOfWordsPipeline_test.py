import unittest
import numpy as np
import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
import bagOfWordsPipeline


class bagOfWordsPipelineTest(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super(bagOfWordsPipelineTest, self).__init__(*args, **kwargs)

        self.bowp = bagOfWordsPipeline.BagOfWordsPipeLine()    

    def assertArrayEqual(self, arr1: np.ndarray, arr2: np.ndarray):
        for d1, d2 in zip(arr1.shape, arr2.shape):
            self.assertEqual(d1, d2)
        for i in range(arr1.size):
            # print(arr1.take(i))
            self.assertEqual(arr1.take(i), arr2.take(i))

    def test_loading(self):
        self.bowp.loadData()

    def test_train(self):
        self.bowp.trainTokenizer()
    
    def test_textsToPaddedSequences(self):
        newBowp = bagOfWordsPipeline.BagOfWordsPipeLine()    
        newBowp.loadData()
        newBowp.trainTokenizer()
        posTextPaddedSequences = newBowp.textsPosToPaddedSequences()
        negTextPaddedSequences = newBowp.textsNegToPaddedSequences()
        print(posTextPaddedSequences)
        print(negTextPaddedSequences)
                
    def test_textsToMatrix(self):
        newBowp = bagOfWordsPipeline.BagOfWordsPipeLine()    
        newBowp.loadData()
        newBowp.trainTokenizer()
        posTextTokenMatrix = newBowp.textsPosToMatrix()
        negTextTokenMatrix = newBowp.textsNegToMatrix()
        print(posTextTokenMatrix)
        print(negTextTokenMatrix)




if __name__ == "__main__":
    unittest.main()