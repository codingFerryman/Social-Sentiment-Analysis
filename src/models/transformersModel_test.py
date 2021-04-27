import unittest
import transformers
import transformersModel
import numpy as np
import tensorflow as tf
import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
import inputFunctions
from preprocessing.pretrainedTransformersPipeline import PretrainedTransformersPipeLine
from models.modelMaps import getModelMapAvailableNames
import loggers

logger = loggers.getLogger("RobertaModelTest", debug=True)


class TransformersModelTest(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super(TransformersModelTest, self).__init__(*args, **kwargs)
        
    
    def test_createModelRoberta(self):
        logger.debug("Testing createModel Roberta")
        roberta = transformersModel.TransformersModel()
        roberta.pipeLine.loadFunction = inputFunctions.loadDataForUnitTesting
        roberta.loadData()
        roberta.createModel()

    def test_createAllTransformerModels(self):
        for name in getModelMapAvailableNames():
            logger.debug(f"Testing createModel for {name}")
            transfModel = transformersModel.TransformersModel(pipeLine={'modelName': name}, modelName=name)
            transfModel.pipeLine.loadFunction = inputFunctions.loadDataForUnitTesting
            transfModel.loadData()
            transfModel.createModel()

    def test_testModel(self):
        logger.debug("Testing testModel")
        roberta = transformersModel.TransformersModel()
        roberta.pipeLine.loadFunction = inputFunctions.loadDataForUnitTesting
        roberta.loadData()
        roberta.registerMetric(tf.keras.metrics.SparseCategoricalAccuracy('accuracy'))
        roberta.testModel(epochs=2)

    def test_testModelAllTransformerModels(self):
        for name in getModelMapAvailableNames():
            logger.debug(f"Testing testModel for {name}")
            transfModel = transformersModel.TransformersModel(pipeLine={'modelName': name}, modelName=name)
            transfModel.pipeLine.loadFunction = inputFunctions.loadDataForUnitTesting
            transfModel.loadData()
            transfModel.registerMetric(tf.keras.metrics.SparseCategoricalAccuracy('accuracy'))
            transfModel.testModel(epochs=2)
        
if __name__ == "__main__":
    unittest.main()