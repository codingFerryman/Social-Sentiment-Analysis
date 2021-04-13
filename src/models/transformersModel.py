import tensorflow as tf
import transformers
from transformers import RobertaConfig
import numpy as np
import sklearn
from sklearn import model_selection
import typing
import os, sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from preprocessing.pretrainedTransformersPipeline import PretrainedTransformersPipeLine
from models.Model import ModelConstruction

import inputFunctions
import loggers
import pdb


logger = loggers.getLogger("RobertaModel", True)

def getDefaultTokenizer(loadFunction=None):
    if loadFunction == None:
        return PretrainedTransformersPipeLine(tokenizer=transformers.RobertaTokenizer, 
                                                        pretrainedTokenizerName='roberta-base')
    else:
        return PretrainedTransformersPipeLine(loadFunction=loadFunction, tokenizer=transformers.RobertaTokenizer, 
                                                        pretrainedTokenizerName='roberta-base')
class TransformersModel(ModelConstruction):
    def __init__(self, dataPath:str=None, pipeLine=None, loadFunction=None, modelName:str="roberta"):
        self.configuration = transformers.RobertaConfig()
        if pipeLine == None:
            self.pipeLine = getDefaultTokenizer(loadFunction=loadFunction)
        else:
            self.pipeLine = pipeLine
        self._registeredMetrics = []
        self._modelName = modelName

    def loadData(self):
        self.pipeLine.loadData()
        # self.pipeLine.trainTokenizer()
        self._dataLoaded = True
    
    def createModel(self, **kwargs) -> tf.keras.Model:
        assert self._dataLoaded, "data should be loaded before calling createModel"
        # assert self.pipeLine.num_words != None, "pipeline should have num_words != None"
        model = transformers.TFRobertaForSequenceClassification.from_pretrained('roberta-base')
        return model

    def testModel(self, train_val_split_iterator: typing.Iterator = [sklearn.model_selection.train_test_split], **kwargs):
        logger.info("Starting testing of RobertaModel")
        num_epochs = kwargs['epochs']
        for i, train_test_split in enumerate(train_val_split_iterator):
            logger.debug(f'{i}-th enumeration of train_val split iterator under cross validation')
            train_dataset, val_dataset = self.pipeLine.getEncodedDataset(None)
            self.model = self.createModel()
            optimizer = tf.keras.optimizers.Adam(learning_rate=3e-5, epsilon=1e-08, clipnorm=1.0)
            loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
            self.model.compile(optimizer=optimizer, loss=loss, metrics=self._registeredMetrics)
            self.model.fit(train_dataset, epochs=num_epochs)

    def getTestResults(self) -> typing.List[dict]:
        """This method gets results from last training

        Returns:
            typing.List[dict]: list of dictionaries containing metric results
        """
        return self.currentResults

    def registerMetric(self, metric: 'tf.keras.metrics.Metric'):
        self._registeredMetrics.append(metric)

    def save(self, model_path: str, model_id: int):
        logger.info("Saving TransformersModel")
        self.model_params["class"] = self.__class__.__name__
        with open(os.path.join(model_path, 'params.json'), 'w') as json_file:
            json.dump(self.model_params, json_file)
        self.model.save(os.path.join(model_path, f'{self._modelName}_{model_id}.h5'))

    @staticmethod
    def load(load_folder_path:str, model_name:str, model_id:int):
        return BaseJointTransformerModel.load_model_by_class(JointTransRobertaModel, load_folder_path, f'{model_name}_{model_id}.h5')