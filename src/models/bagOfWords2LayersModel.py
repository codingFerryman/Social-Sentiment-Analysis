import tensorflow as tf
import numpy as np
import sklearn
from sklearn import model_selection
import typing
import os, sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from preprocessing.bagOfWordsPipeline import BagOfWordsPipeLine
from models.Model import ModelConstruction
import inputFunctions
import loggers

logger = loggers.getLogger("BagOfWordsModel", True)

class BagOfWords2LayerModel(ModelConstruction):
    def __init__(self, dataPath:str=None, pipeLine=BagOfWordsPipeLine()):
        logger.info("BagOfWordsModel created")
        self.dataPath = dataPath
        self._model = None
        self.all_data = []
        self.pipeLine = pipeLine
        self.paddedSequencesPos = [] 
        self.paddedSequencesNeg = []
        self._registeredMetrics = []
        self._dataLoaded = False
    
    def loadData(self):
        self.pipeLine.loadData()
        self.pipeLine.trainTokenizer()
        self.paddedSequencesPos = self.pipeLine.textsPosToPaddedSequences()
        self.paddedSequencesNeg = self.pipeLine.textsNegToPaddedSequences()
        self.all_data = self.pipeLine.mixPositiveNegativeWithLabels(
                                        self.paddedSequencesPos,
                                        self.paddedSequencesNeg,
                                        posLabel=0.99,
                                        negLabel=0)
        
        self._dataLoaded = True
        logger.debug(self.paddedSequencesNeg)

    def createModel(self, **kwargs) -> tf.keras.Model:
        assert self._dataLoaded, "data should be loaded before calling createModel"
        assert self.pipeLine.num_words != None, "pipeline should have num_words != None"
        logger.info("Creating Model")
        model = tf.keras.Sequential()
        # Embedding layer
        embedding_layer = tf.keras.layers.Embedding(
            input_dim=self.pipeLine.num_words + 1,
            output_dim=kwargs["embedding_output_dim"],
            input_length=kwargs["max_sequence_length"],
            trainable=kwargs['embedding_trainable'])
        model.add(embedding_layer)
        model.add(tf.keras.layers.Flatten())
        model.add(tf.keras.layers.Dense(kwargs["layer_size"], activation='relu'))
        model.add(tf.keras.layers.Dense(1))
        return model

    def trainModel(self, train_data: typing.Tuple[list, np.ndarray], val_data: typing.Tuple[list, np.ndarray]=[], trainable=True, **kwargs):
        assert self._dataLoaded, "Data was not loaded before launching training"
        logger.info("fetching data for training")
        trainSequences, y = train_data
        encData = tf.data.Dataset.from_tensor_slices((
            trainSequences,
            list(y),)).shuffle(1000).batch(batch_size=32)
        max_sequence_length = trainSequences[0].shape[0]
        logger.debug(f"sequence length= {max_sequence_length}")
        self._model = self.createModel(max_sequence_length=max_sequence_length,
                                       embedding_output_dim=32,
                                       embedding_trainable=trainable,
                                       layer_size=32,
                                       **kwargs)
        
        self._model.compile(optimizer=kwargs['optimizer'], loss=kwargs['loss'], metrics=self._registeredMetrics)
        logger.info("Starting model training")
        self._model.fit(encData)
        logger.info("Model training finished")
        

    def testModel(self, train_test_split_iterator: typing.Iterator = [sklearn.model_selection.train_test_split], **kwargs):
        logger.info("Starting testing of Model")
        iteratorConfig = {}
        if "iterator" in kwargs.keys():
            iteratorConfig = kwargs["iterator"]
        trainingsResults = [] 
        for i, itSplitter in enumerate(train_test_split_iterator):
            logger.info(f"Model test iteration {i}")
            train_dataX, test_dataX, train_datay, test_datay = itSplitter(*self.all_data, test_size= kwargs["test_size"], **iteratorConfig)
            l1, l2 = len(train_dataX), len(train_datay)
            logger.debug(f"{l1}, {l2}")
            
            self.trainModel((train_dataX, train_datay), **kwargs)
            res = self._model.evaluate(test_dataX, test_datay, return_dict=True)
            trainingsResults.append(res)
        self.currentResults = trainingsResults

    def getTestResults(self) -> typing.List[dict]:
        """This method gets results from last training

        Returns:
            typing.List[dict]: list of dictionaries containing metric results
        """
        return self.currentResults

    def registerMetric(self, metric: 'tf.keras.metrics.Metric'):
        self._registeredMetrics.append(metric)
    