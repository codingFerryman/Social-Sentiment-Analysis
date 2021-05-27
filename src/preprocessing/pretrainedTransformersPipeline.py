import tensorflow as tf
import torch
import transformers
import os, sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from preprocessing import InputPipeline
import inputFunctions
import loggers
import numpy as np
import typing
from enum import Enum
logger = loggers.getLogger("PretrainedTransformersPipeLine", debug=True)
import pdb

class torchOrTFEnum(Enum):
    """ Enumeration to pick between the tensorflow interface and pytorch
    """ 
    TF = 0
    TORCH = 1

class TwitterDatasetTorch(torch.utils.data.Dataset):
    def __init__(self, encodings:dict, labels:list, shuffle:bool=False, shufflingParameter:int=-1):
        """ This is a wrapper to pass data of the twitter dataset to pytorch.
        Args:
            encodings (dict): The encodings is a dictionary which is turned from a list of the tokenized tweet texts.
            labels (list): The labels from the dataset.
            shuffle (bool): To shuffle or not shuffle data. Default False = does not shuffle.
            shufflingParameter (int): The parameter to initialize the shuffling with.

        """
        self.encodings = encodings
        self.labels = torch.from_numpy(np.array(labels)).long()

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)



class PretrainedTransformersPipeLine(InputPipeline.InputPipeline):
    def __init__(self, dataPath:str=None, loadFunction:callable=None, tokenizer=None, pretrainedTokenizerName:str=''):
        """ This class accepts tokenizers from the transformers library and wraps their functionality, so that every tokenizer is used
        the same way and can be also used from the model wrapper for the pretrainedTransformersModel. Most tokenizers are pretrained to
        other datasets such as wikipedia. To see all available tokenizers see the file: ./pipelineMaps.py .

        Args:
            dataPath (str): The path of the tweet dataset. Defaults to None, where the default path of the load function is used.
            loadFunction (callable): The function to load the tweet texts from the dataset.
            tokenizer: tokenizer used from the transformers library. Defaults to None which uses the 'bert-base-uncased'. 
            pretrainedTokenizerName (str): The name of the tokenizer if this is supplied explicitely. Defaults to ''.

        """
        
        logger.info("PretrainedTransformersPipeLine created")
        self.dataPath = dataPath        
        self.allData = []
        self.dataPos = []
        self.dataNeg = []
        self.num_words:int = None
        # if no lad function is set, then the default load data is
        # used which loads all data
        if loadFunction == None:
            self.loadFunction = inputFunctions.loadData
        else:
            self.loadFunction = loadFunction
        # If no tokenizer is used then 
        # bert tokenizer will be used. The Bert tokenizer is already pretrained
        if tokenizer == None:
            self._tokenizer = transformers.BertTokenizer.from_pretrained('bert-base-uncased')
        else:
            if pretrainedTokenizerName != '':
                self._tokenizer = tokenizer.from_pretrained(pretrainedTokenizerName)
            else:
                self._tokenizer = tokenizer
        # The pretrained tokenizer name helps to identify which tokenizer is used when logging messages
        # while training or using this pipeline from a model wrapper
        if pretrainedTokenizerName != '':
            self._pretrainedTokenizerName = self._tokenizer.pretrained_init_configuration.get('pretrained_model_name_or_path')
        else:
            self._pretrainedTokenizerName = pretrainedTokenizerName
        self._dataLoaded = False

    def loadData(self):
        """ This loads the data using the loadFunction supplied by the inputFunctions in the constructor.
        It receives the data separately and mixes them together afterwards. Because for training these tokenizers dont care about
        the final labeling.
        """
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
                                truncation=True, padding=False, return_tensors="pt")
        return ret
        
    def textsToPaddedSequences(self, texts: list, length:int=-1):
        logger.info(f"transforming texts to padded sequences with PretrainedTransformersPipeLine {self._pretrainedTokenizerName}")
        # try:
        if length == -1:
            ret = self._tokenizer(texts, add_special_tokens=True, 
                                truncation=True, padding='longest', return_tensors="pt")
        else:
            ret = self._tokenizer(texts, add_special_tokens=True, 
                                truncation=True, padding='longest', max_length=length, return_tensors="pt")
        # except:
        #     ret = self._tokenizer(texts)
        #     # if length == -1:

        #     # else:
        #     #     ret = self._tokenizer(texts)
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

    def argmixPositiveNegative(self, textsPos:list, textsNeg:list) -> np.ndarray:
        """

        Args:
            textsPos (list[str]): List of strings with the positive labeled tweet texts.
            textsNeg (list[str]): List of strings with the negative labeled tweet texts.

        Returns:
            np.ndarray: A randomly shuffled list with the the labels of positive text (=1) and negative text (=0). 

        """
        
        negAsZeros = np.zeros((len(textsNeg),), dtype=np.int32)
        posAsOnes = np.ones((len(textsPos),), dtype=np.int32)
        concatenated = np.concatenate((negAsZeros, posAsOnes))
        np.random.shuffle(concatenated) # does not use more memory
        return concatenated
    
    def getLabels(self, argMix:list=[], posList:list=[], negList:list=[], posLabel:int=1, negLabel:int=-1) -> np.ndarray:
        """ This returns the renewed labels for the positive and negative tweets randomly shuffled accoording to argMix argument.
        TODO: Explain better.
        Args:
            argMix (list): list of mixed labels.
            posList (list): list of strings of positive tweets.
            negList (list): list of strings of negative tweets.
            posLabel (int): positive label values. Defaults to 1.
            negLabel (int): negative label value. Defaults to -1.

        Returns:
           (np.ndarray): The labels for the the positive and negative tweets concatenated in the same order as they appear in argMix. 

        """
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
        """

        Returns:
            tuple(int,int,list): text's minimum length in words, texts' maximum length in words, texts with zero length
        Raises:
            Exception: In case no data has been loaded
        """
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
    
    def getEncodedDatasetTF(self, train_dataX:list, train_datay:list, val_dataX:list, val_datay:list, max_len:int, shufflingParameter:int, batch_size:int):
        encDataTrain = tf.data.Dataset.from_tensor_slices((
            dict(self.textsToPaddedSequences(train_dataX, length=max_len)),
            list(train_datay))).shuffle(shufflingParameter).batch(batch_size=batch_size)
        encDataVal = tf.data.Dataset.from_tensor_slices((
            dict(self.textsToPaddedSequences(val_dataX, length=max_len)),
            list(val_datay))).shuffle(shufflingParameter).batch(batch_size=batch_size)
        
        return encDataTrain, encDataVal


    def getEncodedDatasetTorch(self, train_dataX:list, train_datay:list, val_dataX:list, val_datay:list, max_len:int, shufflingParameter:int, batch_size:int):

        encDataTrain = TwitterDatasetTorch(dict(self.textsToPaddedSequences(train_dataX, length=max_len)), train_datay, True, shufflingParameter)
        encDataVal = TwitterDatasetTorch(dict(self.textsToPaddedSequences(val_dataX, length=max_len)), val_datay, True, shufflingParameter)
        
        return encDataTrain, encDataVal

    def getClassWeight(self, posLabel=1, negLabel=0) -> typing.Dict[int, float]:
        assert self._dataLoaded, "Data should be loaded to get the encoded dataset"
        numNeg = np.size(self.dataNeg)
        numPos = np.size(self.dataPos)
        numTotal = numNeg + numPos
        ret = {
            posLabel: numPos/(numTotal),
            negLabel: numNeg/(numTotal)
        }
        return ret
    
    def getEncodedDataset(self, splitter:typing.Callable=None, posLabel=1, negLabel=0, shufflingParameter:int=1000, batch_size:int=64, tfOrPyTorch:torchOrTFEnum=torchOrTFEnum.TF,**splitterConfig):
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
        
        self.splitter = splitter
        
        if splitter == None:
            # tokenLists = self.textsToPaddedSequences(self.allData, max_len)
            logger.debug("No splitter specified")
            if tfOrPyTorch == torchOrTFEnum.TF:
                encDataTrain, encDataVal = self.getEncodedDatasetTF(train_dataX=self.allData, train_datay=list(labels),
                                                                    val_dataX={}, val_datay=[], max_len=max_len, shufflingParameter=shufflingParameter, batch_size=batch_size)
            else:
                encDataTrain, encDataVal = self.getEncodedDatasetTorch(train_dataX=self.allData, train_datay=list(labels),
                                                                    val_dataX={}, val_datay=[], max_len=max_len, shufflingParameter=shufflingParameter, batch_size=batch_size)
            yield (encDataTrain, encDataVal)
        else:
            for train_index, val_index in self.splitter.split(self.allData, labels, **splitterConfig):
                train_dataX, val_dataX = [self.allData[i] for i in train_index], [self.allData[i] for i in val_index]
                train_datay, val_datay = labels[train_index], labels[val_index]
                if tfOrPyTorch == torchOrTFEnum.TF:
                    encDataTrain, encDataVal = self.getEncodedDatasetTF(train_dataX=train_dataX, train_datay=list(train_datay),
                                                                        val_dataX=val_dataX, val_datay=list(val_datay), max_len=max_len, shufflingParameter=shufflingParameter, batch_size=batch_size)
                else:                
                    encDataTrain, encDataVal = self.getEncodedDatasetTorch(train_dataX=train_dataX, train_datay=list(train_datay),
                                                                        val_dataX=val_dataX, val_datay=list(val_datay), max_len=max_len, shufflingParameter=shufflingParameter, batch_size=batch_size)
                yield (encDataTrain, encDataVal)
