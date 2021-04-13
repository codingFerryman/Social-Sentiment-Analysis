import unittest
import robertaModel
import numpy as np
import tensorflow as tf
import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
import inputFunctions
from preprocessing import pretrainedTransformersPipeline

import loggers

logger = loggers.getLogger("RobertaModelTest", debug=True)


class RobertaModelTest(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super(RobertaModelTest, self).__init__(*args, **kwargs)
        
    
    def test_createModel(self):
        logger.debug("Testing createModel")
        roberta = robertaModel.RobertaModel()
        roberta.pipeLine.loadFunction = inputFunctions.loadDataForUnitTesting
        roberta.loadData()
        roberta.createModel()

    def test_testModel(self):
        logger.debug("Testing testModel")
        roberta = robertaModel.RobertaModel()
        roberta.pipeLine.loadFunction = inputFunctions.loadDataForUnitTesting
        roberta.loadData()
        roberta.registerMetric(tf.keras.metrics.SparseCategoricalAccuracy('accuracy'))
        roberta.testModel(epochs=2)

if __name__ == "__main__":
    unittest.main()