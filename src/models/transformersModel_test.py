import os
import unittest

from models.transformersModel import TransformersModel
from utils import loggers, inputFunctions

logger = loggers.getLogger("RobertaModelTest", debug=True)

_current_path = os.path.dirname(os.path.realpath(__file__))


class TransformersModelTest(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super(TransformersModelTest, self).__init__(*args, **kwargs)

    def test_createModelRoberta(self):
        logger.debug("Testing createModel Roberta")
        roberta = TransformersModel(loadFunction=inputFunctions.loadDataForUnitTesting)
        roberta.loadData()
        roberta.createModel()

    def test_createAllTransformerModels(self):

        # Please do NOT test large models here or you will waste lots of time
        with open(os.path.join(_current_path, 'transformersModelNames.txt'), 'r') as fp:
            names = fp.read().splitlines()
        for name in names:
            logger.debug(f"Testing createModel for {name}")
            transfModel = TransformersModel(modelName_or_pipeLine=name,
                                            loadFunction=inputFunctions.loadDataForUnitTesting)
            transfModel.loadData()
            transfModel.createModel()

    def test_testModel(self):
        logger.debug("Testing testModel")
        roberta = TransformersModel(modelName_or_pipeLine='roberta-base')
        roberta.loadData(ratio=0.0001)
        metric = ['glue', 'mrpc']
        roberta.registerMetric(*metric)
        roberta.trainModel(epochs=1, batch_size=64)

    def test_testModelAllTransformerModels(self):
        # Please do NOT test large models here or you will waste lots of time
        with open(os.path.join(_current_path, 'transformersModelNames.txt'), 'r') as fp:
            names = fp.read().splitlines()
        for name in names:
            logger.debug(f"Testing testModel for {name}")
            transfModel = TransformersModel(modelName_or_pipeLine=name,
                                            loadFunction=inputFunctions.loadDataForUnitTesting)
            transfModel.loadData()
            # transfModel.registerMetric({'name': 'accuracy'})
            transfModel.trainModel(epochs=2, batch_size=64, weight_decay=0.01, warmup_steps=10)


if __name__ == "__main__":
    unittest.main()
