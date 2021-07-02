import os
import sys

import tensorflow as tf

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from preprocessing import InputPipeline
from utils import inputFunctions, loggers
import numpy as np
import typing

logger = loggers.getLogger("BagOfWordsPipeline", debug=True)

class BagOfWordsPipeLine(InputPipeline.InputPipeline):
    def __init__(self, dataPath=None, loadFunction:callable=None):
        logger.info("BagOfWordsPipeline created")
        self.dataPath = dataPath
        self._tokenizer = None
        self.allData = []
        self.dataPos = []
        self.dataNeg = []
        self.num_words:int = None
        if loadFunction == None:
            self.loadFunction = inputFunctions.loadData
        else:
            self.loadFunction = loadFunction
    
    def loadData(self):
        train_pos, train_neg, test_data = self.loadFunction(self.dataPath)
        self.dataPos = train_pos
        self.dataNeg = train_neg
        self.allData = train_pos + train_neg

    def trainTokenizer(self):
        assert self.allData != [], "no data to train"
        logger.info("Creating bag of words tokenizer")
        self._tokenizer = tf.keras.preprocessing.text.Tokenizer(num_words=None,
                                filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n',
                                lower=True, split=' ', char_level=False, oov_token=None,
                                document_count=0)
        logger.info("Starting training of bag of words tokenizer")
        self._tokenizer.fit_on_texts(self.allData)
        logger.info("Finished training of bag of words tokenizer")
        # logger.debug(self._tokenizer.get_config())
        self.num_words = len(self._tokenizer.get_config()['word_counts'])


    def textsToSequences(self, texts: list) -> tf.Tensor:
        return self._tokenizer.texts_to_sequences(texts)
        
    def textsToPaddedSequences(self, texts: list):
        logger.info("transforming texts to padded sequences with bag of words tokenizer")
        sequences = self.textsToSequences(texts)
        return tf.keras.preprocessing.sequence.pad_sequences(sequences, padding='post', maxlen=69)
    
    def textsToMatrix(self, texts: list) -> tf.Tensor:
        logger.info("transforming texts to matrix with bag of words tokenizer")
        sequencesPos = self.textsToSequences(texts) 
        # paddedSequences = tf.keras.preprocessing.sequence.pad_sequences(sequencesPos, padding='post')
        return self._tokenizer.sequences_to_matrix(sequencesPos)


    def textsPosToPaddedSequences(self):
        return self.textsToPaddedSequences(self.dataPos)

    def textsNegToPaddedSequences(self):
        return self.textsToPaddedSequences(self.dataNeg)

    def textsPosToMatrix(self):
        return self.textsToMatrix(self.dataPos)

    def textsNegToMatrix(self):
        return self.textsToMatrix(self.dataNeg)

    def argmixPositiveNegative(self, textsPos:list, textsNeg:list) -> list:
        negAsZeros = np.zeros((len(textsNeg),), dtype=np.int32)
        posAsOnes = np.ones((len(textsPos),), dtype=np.int32)
        concatenated = np.concatenate((negAsZeros, posAsOnes))
        np.random.shuffle(concatenated) # does not use more memory
        return concatenated
    
    def getLabels(self, argMix:list=[], posList:list=[], negList:list=[], posLabel:int=1, negLabel:int=-1) -> np.ndarray:
        # y = pipeline.getLabels(...)
        assert ((argMix == []) != (posList == [] and negList == [])),\
            "argMix should be == [] if posList and negList != [] else argMix should be != [] and posList, negList == []"
        if argMix == []:
            posLabels = posLabel * np.ones_like(posList, dtype=np.int32)
            negLabels = negLabel * np.ones_like(negList, dtype=np.int32)
            y = np.concatenate((posLabel, negLabels))
        else:
            oldNegLabel, oldPosLabel = np.sort(np.unique(argMix))
            y = []
            for w in argMix:
                if w == oldNegLabel:
                    y.append(negLabel)
                else:
                    y.append(posLabel)
            y = np.array(y, dtype=np.int32)
        return y

    def mixPositiveNegativeWithLabels(self, textsPos:list, textsNeg:list, posLabel=1, negLabel=0) -> typing.Tuple[list, np.ndarray]:
        argMix = self.argmixPositiveNegative(textsPos, textsNeg)
        finalList = []
        iP = 0
        iN = 0
        for w in argMix: 
            # w = 0 for textsNeg
            # w = 1 for textsPos
            if w == 0:
                finalList.append(textsNeg[iN])
                iN = iN + 1
            else:
                finalList.append(textsPos[iP])
                iP = iP + 1
        labels = self.getLabels(argMix=argMix, posLabel=posLabel, negLabel=negLabel)
        return finalList, labels
