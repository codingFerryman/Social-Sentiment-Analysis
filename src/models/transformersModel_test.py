import os
import sys
import unittest

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
import loggers

logger = loggers.getLogger("RobertaModelTest", debug=True)


class TransformersModelTest(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super(TransformersModelTest, self).__init__(*args, **kwargs)
        
    
    # def test_createModelRoberta(self):
    #     logger.debug("Testing createModel Roberta")
    #     roberta = transformersModel.TransformersModel(loadFunction=inputFunctions.loadDataForUnitTesting)
    #     roberta.loadData()
    #     roberta.createModel()

    # def test_createAllTransformerModels(self):
    #     for name in getModelMapAvailableNames():
    #         logger.debug(f"Testing createModel for {name}")
    #         transfModel = transformersModel.TransformersModel(pipeLine={'modelName': name}, modelName=name, loadFunction=inputFunctions.loadDataForUnitTesting)
    #         transfModel.pipeLine.loadFunction = 
    #         transfModel.loadData()
    #         transfModel.createModel()

    # def test_testModel(self):
    #     logger.debug("Testing testModel")
    #     roberta = transformersModel.TransformersModel()
    #     roberta.loadData()
    #     # roberta.registerMetric({'name': 'accuracy'})
    #     roberta.testModel(epochs=2, batch_size=32, weight_decay=0.01, warmup_steps=10)

    # def test_testModelAllTransformerModels(self):
    #     for name in getModelMapAvailableNames():
    #         logger.debug(f"Testing testModel for {name}")
    #         transfModel = transformersModel.TransformersModel(pipeLine={'modelName': name}, modelName=name, loadFunction=inputFunctions.loadDataForUnitTesting)
    #         transfModel.loadData()
    #         # transfModel.registerMetric({'name': 'accuracy'})
    #         transfModel.trainModel(epochs=2, batch_size=64, weight_decay=0.01, warmup_steps=10)

if __name__ == "__main__":
    unittest.main()
