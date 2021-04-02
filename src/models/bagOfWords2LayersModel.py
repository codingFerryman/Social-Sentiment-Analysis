import tensorflow as tf
from preprocessing.bagOfWordsPipeline import BagOfWordsPipeLine
import inputFunctions
import loggers

logger = loggers.getLogger("BagOfWordsModel", True)

class BagOfWords2LayerModel:
    def __init__(self, dataPath:str=None, pipeLine=BagOfWordsPipeLine):
        logger.info("BagOfWordsModel created")
        self.dataPath = dataPath
        self._model = None
        self.train_data = []
        self.pipeLine = pipeLine
        self.paddedSequencesPos = [] 
        self.paddedSequencesNeg = [] 
        self._dataLoaded = False

    def loadData(self):
        self.pipeLine.loadData()
        self.pipeLine.trainTokenizer()
        self.paddedSequencesPos = self.pipeLine.textsPosToPaddedSequences()
        self.paddedSequencesNeg = self.pipeLine.textsNegToPaddedSequences()
        self._dataLoaded = True

    def createModel(self):
        
    def trainModel(self):
        assert self.dataLoaded, "Data was not loaded before launching training"
        self._model = self.createModel()


    def test(self):


    def registerMetric(self):
        
