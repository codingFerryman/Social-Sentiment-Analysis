import tensorflow as tf
import transformers
import os, sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from preprocessing import InputPipeline
import inputFunctions
import loggers
import numpy as np
import typing
import pdb

logger = loggers.getLogger("PretrainedTransformersPipeLine", debug=True)

class PretrainedTransformersPipeLine(InputPipeline.InputPipeline):
    def __init__(self, dataPath=None, loadFunction:callable=None, tokenizer=None, pretrainedTokenizerName='bert-base-uncased'):
        logger.info("PretrainedTransformersPipeLine created")
        self.dataPath = dataPath        
        self.allData = []
        self.dataPos = []
        self.dataNeg = []
        self.num_words:int = None
        if loadFunction == None:
            self.loadFunction = inputFunctions.loadData
        else:
            self.loadFunction = loadFunction
        self._pretrainedTokenizerName = pretrainedTokenizerName
        if tokenizer == None:
            self._tokenizer = transformers.BertTokenizer.from_pretrained('bert-base-uncased')
        else:
            self._tokenizer = tokenizer.from_pretrained(pretrainedTokenizerName)
        self._dataLoaded = False

    def loadData(self):
        logger.info(f"loading data for PretrainedTransformersPipeLine {self._pretrainedTokenizerName}")
        train_pos, train_neg, test_data = self.loadFunction(self.dataPath)
        self.dataPos = train_pos
        self.dataNeg = train_neg
        self.allData = train_pos + train_neg
        self._dataLoaded = True

    def trainTokenizer(self):
        assert self.allData != [], "no data to train"
        logger.info(f"No train phase in PretrainedTransformersPipeLine {self._pretrainedTokenizerName}")
        

    def textsToSequences(self, texts: list) -> tf.Tensor:
        ret = self._tokenizer(texts, add_special_tokens=True, 
                                truncation=True, padding=False)
        return ret
        
    def textsToPaddedSequences(self, texts: list, length:int=-1):
        logger.info(f"transforming texts to padded sequences with PretrainedTransformersPipeLine {self._pretrainedTokenizerName}")
        if length == -1:
            ret = self._tokenizer(texts, add_special_tokens=True, 
                                truncation=True, padding='longest')
        else:
            ret = self._tokenizer(texts, add_special_tokens=True, 
                                truncation=True, padding='longest', max_length=length)
        return ret
    
    def textsToMatrix(self, texts: list) -> tf.Tensor:
        logger.info(f"transforming texts to matrix with PretrainedTransformersPipeLine {self._pretrainedTokenizerName}")
        sequencesPos = self.textsToSequences(texts) 
        # paddedSequences = tf.keras.preprocessing.sequence.pad_sequences(sequencesPos, padding='post')
        return self._tokenizer.sequences_to_matrix(sequencesPos)


    def textsPosToPaddedSequences(self, length:int=-1):
        return self.textsToPaddedSequences(self.dataPos, length)

    def textsNegToPaddedSequences(self, length:int=-1):
        return self.textsToPaddedSequences(self.dataNeg, length)

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
        assert ((len(argMix) == 0) != (len(posList) == len(negList) == 0)),\
            "argMix should be == [] if posList and negList != [] else argMix should be != [] and posList, negList == []"
        if len(argMix) == 0:
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
    
    def getSequenceMaxLength(self) -> typing.Tuple[int,int,list]:
        assert self._dataLoaded, "Data should be loaded to get sequences max length"
        # The maximum, minimum number of words in tweets
        # ... and empty entries
        if len(self.allData) <= 0:
            logger.warning("No proper data loading")
            raise Exception("No proper data loading len(self.allData) = 0")
        min_len = len(self.allData[0])
        max_len = 0
        zero_len_idx = []
        for idx, t in enumerate(self.allData):
            t_len = len(t.split())
            if t_len == 0:
                zero_len_idx.append(idx)
            if t_len > max_len:
                max_len = t_len
            if t_len < min_len:
                min_len = t_len

        return min_len, max_len, zero_len_idx
    
    def getEncodedDataset(self, splitter:typing.Callable=None, posLabel=1, negLabel=0, **splitterConfig):
        assert self._dataLoaded, "Data should be loaded to get the encoded dataset"
        # create labels
        negAsZeros = np.zeros((len(self.dataNeg),), dtype=np.int32)
        posAsOnes = np.ones((len(self.dataPos),), dtype=np.int32)
        argMix = np.concatenate((posAsOnes, negAsZeros))
        labels = self.getLabels(argMix=argMix, posLabel=posLabel, negLabel=negLabel)
        # get max sequence length
        min_len, max_len, zero_len_idx = self.getSequenceMaxLength()
        # if min length == 0 delete the empty texts
        if min_len == 0:
            logger.debug('Deleting zero length texts and labels because min_len = 0')
            self.allData = [d for i,d in enumerate(self.allData) if not(i in zero_len_idx)]
            labels = [l for i,l in enumerate(labels) if not(i in zero_len_idx)]
        if splitter == None:
            tokenLists = self.textsToPaddedSequences(self.allData, max_len)
            encDataTrain = tf.data.Dataset.from_tensor_slices((
                dict(tokenLists),
                list(labels))).shuffle(1000).batch(batch_size=32)
            encDataVal = tf.data.Dataset.from_tensor_slices((
                {},
                [])).shuffle(1000).batch(batch_size=32)
        else:
            train_dataX, test_dataX, train_datay, test_datay = splitter(self.allData, labels, **splitterConfig)
            
            encDataTrain = tf.data.Dataset.from_tensor_slices((
                dict(self.textsToPaddedSequences(train_dataX, length=max_len)),
                list(train_datay))).shuffle(1000).batch(batch_size=32)
            encDataVal = tf.data.Dataset.from_tensor_slices((
                dict(self.textsToPaddedSequences(test_dataX, length=max_len)),
                list(test_datay))).shuffle(1000).batch(batch_size=32)
        
        return encDataTrain, encDataVal