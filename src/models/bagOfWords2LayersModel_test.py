import unittest
import bagOfWords2LayersModel

class BagOfWords2LayersModel(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super(BagOfWords2LayersModel, self).__init__(*args, **kwargs)
        self.bowm = bagOfWords2LayersModel.BagOfWords2LayerModel()
        
    def assertArrayEqual(self, arr1: np.ndarray, arr2: np.ndarray):
        for d1, d2 in zip(arr1.shape, arr2.shape):
            self.assertEqual(d1, d2)
        for i in range(arr1.size):
            # print(arr1.take(i))
            self.assertEqual(arr1.take(i), arr2.take(i))
    
if __name__ == "__main__":
    unittest.main()