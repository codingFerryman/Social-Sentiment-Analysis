import unittest
import numpy as np
import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
import bagOfWordsPipeline


class bagOfWordsPipelineTest(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super(bagOfWordsPipelineTest, self).__init__(*args, **kwargs)

    def assertArrayEqual(self, arr1: np.ndarray, arr2: np.ndarray):
        for d1, d2 in zip(arr1.shape, arr2.shape):
            self.assertEqual(d1, d2)
        for i in range(arr1.size):
            # print(arr1.take(i))
            self.assertEqual(arr1.take(i), arr2.take(i))

    def test_loading(self):
        newBowp = bagOfWordsPipeline.BagOfWordsPipeLine()
        newBowp.loadData()

    def test_train(self):
        newBowp = bagOfWordsPipeline.BagOfWordsPipeLine()  
        newBowp.loadData()
        newBowp.trainTokenizer()
        print(newBowp.num_words)        

    def test_textsToPaddedSequences(self):
        newBowp = bagOfWordsPipeline.BagOfWordsPipeLine()    
        newBowp.loadData()
        newBowp.trainTokenizer()
        posTextPaddedSequences = newBowp.textsPosToPaddedSequences()
        negTextPaddedSequences = newBowp.textsNegToPaddedSequences()
        print(posTextPaddedSequences)
        print(negTextPaddedSequences)

    # def test_textsToMatrix(self):
    #     newBowp = bagOfWordsPipeline.BagOfWordsPipeLine()    
    #     newBowp.loadData()
    #     newBowp.trainTokenizer()
    #     posTextTokenMatrix = newBowp.textsPosToMatrix()
    #     negTextTokenMatrix = newBowp.textsNegToMatrix()
    #     print(posTextTokenMatrix)
    #     print(negTextTokenMatrix)
    
    def test_argmixPositiveNegative(self):
        newBowp = bagOfWordsPipeline.BagOfWordsPipeLine()    
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
    
    
    def test_mixPositiveNegativeWithLabels(self):
        newBowp = bagOfWordsPipeline.BagOfWordsPipeLine()    
        newBowp.loadData()
        newBowp.trainTokenizer()
        posTextPaddedSequences = newBowp.textsPosToPaddedSequences()
        negTextPaddedSequences = newBowp.textsNegToPaddedSequences()
        res = newBowp.mixPositiveNegativeWithLabels(posTextPaddedSequences, negTextPaddedSequences)
        # print(res.shape)
if __name__ == "__main__":
    unittest.main()