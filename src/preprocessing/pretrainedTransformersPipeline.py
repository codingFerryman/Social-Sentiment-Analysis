import re
import random
from typing import Tuple, Dict, Callable

import numpy as np
import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer
from transformers.file_utils import PaddingStrategy
from transformers.tokenization_utils_base import PreTrainedTokenizerBase

from preprocessing.InputPipeline import InputPipeline
from utils.inputFunctions import loadData
from utils.loggers import getLogger

logger = getLogger("PretrainedTransformersPipeLine", debug=True)


class TwitterDatasetTorch(Dataset):
    def __init__(self, text: list, labels: list, tokenizer: PreTrainedTokenizerBase, max_length: int = None,
                 tokenizerConfig: dict = None):
        """ This is a wrapper to pass data of the twitter dataset to pytorch.
        Args:
            text (list): The list of raw text.
            labels (list): The labels from the dataset.
            tokenizer (PreTrainedTokenizerBase): The pretrained tokenizer from transformers
            max_length (int): The max length of the padding/truncation
            tokenizerConfig (dict): A dictionary with tokenizers configuration
        """
        self.text_list = text
        self.labels = labels
        self.tokenizer = tokenizer
        if tokenizerConfig is None:
            tokenizerConfig = {"padding": PaddingStrategy.MAX_LENGTH}
        if max_length is None:
            tokenizerConfig["max_length"] = 256
        else:
            tokenizerConfig["max_length"] = max_length
            tokenizerConfig["truncation"] = True

        self.tokenizerConfig = tokenizerConfig

    def __getitem__(self, idx):
        tweet = self.text_list[idx]
        tweet = self.cleaning(tweet)
        inputs = self.tokenizer.encode_plus(
            text=tweet,
            **self.tokenizerConfig
        )
        _ids = torch.tensor(inputs['input_ids'], dtype=torch.int)
        _mask = torch.tensor(inputs['attention_mask'], dtype=torch.uint8)
        _label = torch.tensor(self.labels[idx], dtype=torch.long)
        return {
            'input_ids': _ids,
            'attention_mask': _mask,
            'labels': _label
        }

    def __len__(self):
        return len(self.labels)

    @staticmethod
    def cleaning(text):
        return re.sub(r'(<.*?>)|(\r\n|\r|\n)|(\'|\")', '', text.lstrip())


class PretrainedTransformersPipeLine(InputPipeline):
    """ This class accepts tokenizers from the transformers library and wraps their functionality, so that every
    tokenizer is used the same way and can be also used from the model wrapper for the pretrainedTransformersModel.
    Most tokenizers are pretrained to other datasets such as wikipedia.
    """

    def __init__(self, model_name_or_path: str = None, dataPath: str = None, loadFunction: callable = None):
        """
        Args:
            model_name_or_path (str): The name of a checkpoint of huggingface transformers model.
            dataPath (str): The path of the tweet dataset. Defaults to None,
            loadFunction (callable): The function to load the tweet texts from the dataset.
        """

        logger.info("PretrainedTransformersPipeLine created")
        self.dataPath = dataPath
        self.allData = []
        self.dataPos = []
        self.dataNeg = []

        if model_name_or_path is None:
            model_name_or_path = 'roberta-base'
        if loadFunction is None:
            self.loadFunction = loadData
        else:
            self.loadFunction = loadFunction
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        self._dataLoaded = False

    def loadData(self, ratio='sub'):
        """ This loads the data using the loadFunction supplied by the inputFunctions in the constructor.
        It receives the data separately and mixes them together afterwards. Because for training these tokenizers
        don't care about the final labeling.
        """
        logger.info(f"loading data for PretrainedTransformersPipeLine {self.tokenizer.name_or_path}")
        train_pos, train_neg, test_data = self.loadFunction(self.dataPath, ratio)
        self.dataPos = train_pos
        self.dataNeg = train_neg
        self.allData = train_pos + train_neg
        self._dataLoaded = True

    def is_data_loaded(self):
        return self._dataLoaded

    def getLabels(self, argMix: list = [], posList: list = [], negList: list = [], posLabel: int = 1,
                  negLabel: int = -1) -> np.ndarray:
        """
        This returns the renewed labels for the positive and negative tweets randomly shuffled
        ... according to argMix argument.
        Args:
            argMix (list): list of mixed labels.
            posList (list): list of strings of positive tweets.
            negList (list): list of strings of negative tweets.
            posLabel (int): positive label values. Defaults to 1.
            negLabel (int): negative label value. Defaults to -1.

        Returns:
           (np.ndarray): The labels for the the positive and negative tweets concatenated in the same order
            ... as they appear in argMix.

        """
        # y = pipeline.getLabels(...)
        assert ((len(argMix) == 0) != (len(posList) == len(negList) == 0)), \
            "argMix should be == [] if posList and negList != [] else argMix should be != [] and posList, negList == []"
        if len(argMix) == 0:
            posLabels = posLabel * np.ones_like(posList, dtype=np.int32)
            negLabels = negLabel * np.ones_like(negList, dtype=np.int32)
            y = np.concatenate((posLabels, negLabels))
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

    def getSequenceMaxLength(self) -> Tuple[int, int, list]:
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

    def getEncodedDatasetTorch(self, train_dataX: list, train_datay: list, val_dataX: list, val_datay: list,
                               max_len: int, tokenizerConfig: dict):
        """Convert the raw texts to PyTorch Datasets for training"""

        encDataTrain = TwitterDatasetTorch(text=train_dataX, labels=train_datay,
                                           tokenizer=self.tokenizer,
                                           max_length=max_len,
                                           tokenizerConfig=tokenizerConfig)
        encDataVal = TwitterDatasetTorch(text=val_dataX, labels=val_datay,
                                         tokenizer=self.tokenizer,
                                         max_length=max_len,
                                         tokenizerConfig=tokenizerConfig)
        return encDataTrain, encDataVal

    def getClassWeight(self, posLabel=1, negLabel=0) -> Dict[int, float]:
        assert self._dataLoaded, "Data should be loaded to get the encoded dataset"
        numNeg = np.size(self.dataNeg)
        numPos = np.size(self.dataPos)
        numTotal = numNeg + numPos
        ret = {
            posLabel: numPos / numTotal,
            negLabel: numNeg / numTotal
        }
        return ret

    def getEncodedDataset(self, splitter: Callable = None,
                          posLabel=1, negLabel=0, stratify=True,
                          tokenizerConfig: dict = None,
                          **splitterConfig):
        """
        Split the training dataset to encoded training and validation datasets.
        Args:
            splitter (Callable): the function to split the dataset
            posLabel (int): the label of positive texts
            negLabel (int): the label of negative texts
            stratify (bool): if stratify then keep the labels balanced during splitting.
            tokenizerConfig (dict): tokenizer configuration; "tokenizer_config" field in the training configuration file
            **splitterConfig: the configuration for the splitter
        """
        assert self._dataLoaded, "Data should be loaded to get the encoded dataset"
        # create labels
        negAsZeros = np.zeros((len(self.dataNeg),), dtype=np.int32)
        posAsOnes = np.ones((len(self.dataPos),), dtype=np.int32)
        argMix = np.concatenate((posAsOnes, negAsZeros))
        labels = list(self.getLabels(argMix=list(argMix), posLabel=posLabel, negLabel=negLabel))
        # get max sequence length
        min_len, max_len, zero_len_idx = self.getSequenceMaxLength()
        # if min length == 0 delete the empty texts
        if min_len == 0:
            logger.debug('Deleting zero length texts and labels because min_len = 0')
            self.allData = [d for i, d in enumerate(self.allData) if not (i in zero_len_idx)]
            labels = [l for i, l in enumerate(labels) if not (i in zero_len_idx)]

        if splitter is None:
            logger.debug("No splitter specified")
            encDataTrain, encDataVal = self.getEncodedDatasetTorch(train_dataX=self.allData,
                                                                   train_datay=labels,
                                                                   val_dataX=[], val_datay=[],
                                                                   max_len=max_len,
                                                                   tokenizerConfig=tokenizerConfig)
            return encDataTrain, encDataVal
        else:
            if stratify:
                stratify_label = labels
            else:
                stratify_label = None
            train_dataX, val_dataX, train_datay, val_datay = splitter(self.allData, labels,
                                                                      stratify=stratify_label,
                                                                      **splitterConfig)
            encDataTrain, encDataVal = self.getEncodedDatasetTorch(train_dataX=train_dataX,
                                                                   train_datay=list(train_datay),
                                                                   val_dataX=val_dataX,
                                                                   val_datay=list(val_datay),
                                                                   max_len=max_len,
                                                                   tokenizerConfig=tokenizerConfig)
            return encDataTrain, encDataVal

    def trainTokenizer(self):
        assert self.allData != [], "no data to train"
        logger.info(f"No train phase in PretrainedTransformersPipeLine {self.tokenizer.name_or_path}")

    def randomizeAllData(self):
        assert self._dataLoaded, "Data not loaded yet"
        _pos = self.dataPos
        _neg = self.dataNeg
        _allData = _pos + _neg
        _allLabels = [1]*len(_pos) + [0]*len(_neg)
        _tmp = list(zip(_allData, _allLabels))
        random.shuffle(_tmp)
        _allData, _allLabels = zip(*_tmp)
        return _allData, _allLabels

    @staticmethod
    def argmixPositiveNegative(textsPos: list, textsNeg: list) -> np.ndarray:
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
        np.random.shuffle(concatenated)  # does not use more memory
        return concatenated

    # def textsToSequences(self, texts: list) -> tf.Tensor:
    #     ret = self._tokenizer(texts, add_special_tokens=True,
    #                           truncation=True, padding=False, return_tensors="pt")
    #     return ret
    #
    # def textsToPaddedSequences(self, texts: list, length: int = -1):
    #     logger.info(
    #         f"transforming texts to padded sequences with PretrainedTransformersPipeLine {self._pretrainedTokenizerName}")
    #     # try:
    #     if length == -1:
    #         ret = self._tokenizer(texts, add_special_tokens=True,
    #                               truncation=True, padding='longest', return_tensors="pt")
    #     else:
    #         ret = self._tokenizer(texts, add_special_tokens=True,
    #                               truncation=True, padding='longest', max_length=length, return_tensors="pt")
    #     # except:
    #     #     ret = self._tokenizer(texts)
    #     #     # if length == -1:
    #
    #     #     # else:
    #     #     #     ret = self._tokenizer(texts)
    #     return ret
    #
    # def textsToMatrix(self, texts: list) -> tf.Tensor:
    #     logger.info(f"transforming texts to matrix with PretrainedTransformersPipeLine {self._pretrainedTokenizerName}")
    #     sequencesPos = self.textsToSequences(texts)
    #     # paddedSequences = tf.keras.preprocessing.sequence.pad_sequences(sequencesPos, padding='post')
    #     return self._tokenizer.sequences_to_matrix(sequencesPos)

    # def textsPosToPaddedSequences(self, length: int = -1):
    #     return self.textsToPaddedSequences(self.dataPos, length)
    #
    # def textsNegToPaddedSequences(self, length: int = -1):
    #     return self.textsToPaddedSequences(self.dataNeg, length)
    #
    # def textsPosToMatrix(self):
    #     return self.textsToMatrix(self.dataPos)
    #
    # def textsNegToMatrix(self):
    #     return self.textsToMatrix(self.dataNeg)
