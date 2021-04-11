import unittest
import bagOfWords2LayersModel
import numpy as np
import tensorflow as tf

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
    
    # def test_loadData(self):
    #     self.bowm.loadData()
    
    # def test_createModel(self):
    #     if not(self.bowm._dataLoaded):
    #         self.bowm.loadData()
    #     model = self.bowm.createModel()
    
    # def test_trainModel(self):
    #     bowm = bagOfWords2LayersModel.BagOfWords2LayerModel()
    #     bowm.loadData()
    #     optimizer = tf.keras.optimizers.Adam(learning_rate=3e-5, epsilon=1e-08, clipnorm=1.0)
    #     loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    #     bowm.trainModel(bowm.all_data, loss=loss, optimizer=optimizer)
    
    def test_default_testModel(self):
        bowm = bagOfWords2LayersModel.BagOfWords2LayerModel()
        bowm.loadData()
        optimizer = tf.keras.optimizers.Adam(learning_rate=3e-5, epsilon=1e-08, clipnorm=1.0)
        loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
        metrics = [tf.keras.metrics.SparseCategoricalAccuracy('accuracy')]
        for m in metrics:
            bowm.registerMetric(m)
        bowm.testModel(loss=loss, optimizer=optimizer, test_size=0.9)
        print(bowm.getTestResults())
if __name__ == "__main__":
    unittest.main()