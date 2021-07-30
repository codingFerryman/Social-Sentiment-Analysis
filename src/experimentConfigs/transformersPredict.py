import json
import os
import random
import sys
from pathlib import Path

import pandas as pd
import torch
from tqdm.auto import tqdm
from transformers import AutoModelForSequenceClassification, AutoTokenizer

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from utils import loggers
from preprocessing.cleaningText import cleaningMap

logger = loggers.getLogger("transformersPredict", True)


class TransformersPredict:
    def __init__(self, load_path:Path, text_path:Path, fast_tokenizer:bool=False, device:str=None, is_test:bool=True, ):
        """ This is the transformers prediction class. It manages the prediction of data given
        a model's checkpoint (load_path) and data to predict (text_path). The device to use can be also
        specified, by default 'cuda:0' is used or 'cpu' (if there is no gpu present).

        Args:
            load_path (Path): The root directory containing 'model' and 'tokenizer'. This is common with the torch checkpoint structure.
            text_path (Path): The text file to be processed. data/test_data.txt by default.
            fast_tokenizer (bool, optional): Whether to use a fast tokenizer. Some models may output an error when this flag is set to false. Defaults to False.
            device (str, optional): String identifier of the device to be used. Defaults to None and 'cuda:0' is used or if there is no gpu present 'cpu' 
            is_test (bool, optional): Whether the data are comming from the test data provided by the kaggle competition or not. Defaults to True.
        """
        self.is_test = is_test

        if device is None:
            if torch.cuda.is_available():
                device = 'cuda:0'
            else:
                device = 'cpu'
        self.device = device
        logger.debug('The program is running on ' + self.device)

        # Load model, tokenizer and their configurations
        model_path = Path(load_path, 'model')
        tokenizer_path = Path(load_path, 'tokenizer')
        config_path = Path(load_path, 'report.json')
        with open(config_path, 'r') as fr:
            cfg = json.load(fr)

        id2label = cfg['model_config']['id2label']
        self.id2label = {int(k): int(v) for k, v in id2label.items()}

        self.tokenizer_config = cfg['tokenizer_config']
        fast_tokenizer_cfg = cfg.get('fast_tokenizer', None)
        if fast_tokenizer_cfg is not None:
            logger.info('fast_tokenizer is overwritten by %s in model config', str(fast_tokenizer_cfg))
            fast_tokenizer = fast_tokenizer_cfg

        text_pre_cleaning = cfg.get('text_pre_cleaning', 'default')
        self.text_pre_cleaning_function = cleaningMap(text_pre_cleaning)

        logger.info(f"Loading model from {load_path}")

        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, use_fast=fast_tokenizer)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_path, output_hidden_states=True,
                                                                        config=cfg['model_config']).to(self.device)

        for param in self.model.parameters():
            param.requires_grad = False

        with open(text_path, 'r') as fp:
            lines = fp.readlines()
        self.data = self.preProcessTest(lines)

        self.text_path = text_path
        self.load_path = load_path

        self.pred = None
        self.pred_scores = None

    def predict(self, batch_size=128):
        """ This is the function that will handle the prediction. It iterates through the provided
        data and uses only the forward part of the models (no training). 
        It stores the data increasingly inside the device of prediction.
        Before using it make sure that there is enough memory in your device to run this script. It
        may exceed memory limits at late stages of prediction and cause prediction stage to fail all
        together.

        Args:
            batch_size (int, optional): The batch size to use for prediction. Defaults to 128.
        """
        predictions = torch.tensor([], dtype=torch.int8)
        scores = torch.tensor([])
        with torch.no_grad():
            for i in tqdm(range(0, len(self.data['text']), batch_size)):
                batch_data = self.data['text'][i:i + batch_size]
                inputs = self.tokenizer(batch_data, return_tensors='pt', **self.tokenizer_config)
                inputs = self.ensure_tensor_on_device(**inputs)
                # Predict
                logit = self.model(**inputs)[0].cpu()
                score = torch.softmax(logit, dim=-1)
                prediction = torch.argmax(score, dim=-1)
                prediction_score = torch.max(score, dim=-1).values
                # Concatenate predictions
                predictions = torch.cat((predictions, prediction), 0)
                scores = torch.cat((scores, prediction_score), 0)
        self.pred = predictions
        self.pred_scores = scores
    
    def predictIterator(self, batch_size=128):
        """

        Args:
            batch_size (int, optional): The batch size to use for prediction. Defaults to 128.

        Yields:
            Tuple: predictions and prediction scores for a batch of text
        """
        with torch.no_grad():
            for i in tqdm(range(0, len(self.data['text']), batch_size)):
                batch_data = self.data['text'][i:i + batch_size]
                inputs = self.tokenizer(batch_data, return_tensors='pt', **self.tokenizer_config)
                inputs = self.ensure_tensor_on_device(**inputs)
                # Predict
                logit = self.model(**inputs)[0].cpu()
                score = torch.softmax(logit, dim=-1)
                prediction = torch.argmax(score, dim=-1)
                prediction_score = torch.max(score, dim=-1).values
                yield prediction, prediction_score

    def extractHiddenStates(self, batch_size=128, appendToList: bool = False):
        """This extract the last hidden state outputs of the model. It can serve as a
        word2vec function.

        Args:
            batch_size (int, optional): The batch size to use for the model forward pass. Defaults to 128.
            appendToList (bool, optional): Whether to append results in a list internally to this object. Defaults to False.

        Yields:
            torch.Tensor: The last hidden state outputs of the model obtained from a series of tweets.
        """
        self.last_hidden_states = []
        with torch.no_grad():
            for i in tqdm(range(0, len(self.data['text']), batch_size)):
                batch_data = self.data['text'][i:i + batch_size]
                inputs = self.tokenizer(batch_data, return_tensors='pt', **self.tokenizer_config)
                inputs = self.ensure_tensor_on_device(**inputs)
                # Predict
                h = self.model(**inputs).hidden_states[-1].cpu()
                self.last_hidden_states.append(h if appendToList else [])
                yield h


    def get_predictions(self):
        pred2label = self.pred.cpu().apply_(self.id2label.get)
        return pred2label

    def getVectorRepresentation(self):
        self.last_hidden_states = [el for el in self.last_hidden_states if el != []]
        return self.last_hidden_states

    def get_scores(self):
        return self.pred_scores

    def submissionToFile(self, save_path:Path=None):
        """ This puts the predictions already predicted to a csv file ready
        for the kaggle api to read and produce the final results from.

        Args:
            save_path (Path, optional): csv file path to put the predictions inside. Defaults to self.laod_path / submission.csv .
        """
        if save_path is None:
            save_path = Path(self.load_path, 'submission.csv')
        logger.info('The submission file will be saved to ' + str(save_path))

        # Format the submission file
        pred_labels = self.get_predictions().tolist()
        id_zero_len = self.data['zero_len_ids']
        pred_id = self.data['ids'] + id_zero_len
        pred_est = pred_labels + [random.choice([-1, 1]) for _ in range(len(id_zero_len))]
        pred_dict = {'Id': pred_id, 'Prediction': pred_est}
        pred_df = pd.DataFrame(pred_dict)
        pred_df = pred_df.astype({
            'Id': int,
            'Prediction': int
        })
        pred_df.sort_values('Id', inplace=True)
        pred_df.to_csv(save_path, index=False)

    def preProcessTest(self, lines: list) -> dict:
        """ This function preprocesses the test dataset.

        Args:
            lines (list): lines of text. Each line contains a tweet

        Returns:
            dict: A dictionary containing
                - text: The preprocessed text
                - ids: The id of the predictions with non empty cleaned text
                - zero_len_ids: The ids having empty cleaned text  (zero length)
        """
        logger.info('Preprocessing the data ...')
        text_id = []
        text = []
        zero_len_idx = []

        if self.is_test:
            # Clean test set
            lines = self.text_pre_cleaning_function(lines, is_test=True)
            for line in lines:
                _text = line.split(',', 1)[-1]
                _id = line.split(',', 1)[0]
                if len(_text) == 0:
                    zero_len_idx.append(_id)
                else:
                    text.append(_text)
                    text_id.append(_id)
        else:
            # Clean the train set
            data = []
            ids = []
            for s in lines:
                _tmp = s.split('\u0001')
                data.append(_tmp[-1])
                ids.append(int(_tmp[0]))
            for idx, sent in tqdm(zip(ids, data)):
                # TODO: Move this cleaning out of the loop to avoid initializing the model every time
                sent_proc = self.text_pre_cleaning_function(sent)
                if len(sent_proc) != 0:
                    text_id.append(idx)
                    text.append(sent_proc)
                else:
                    zero_len_idx.append(idx)
        logger.info('Preprocessed!')
        return {
            'text': text,
            'ids': text_id,
            'zero_len_ids': zero_len_idx
        }

    def ensure_tensor_on_device(self, **inputs):
        """
        Ensure PyTorch tensors are on the specified device.
        Args:
            inputs (keyword arguments that should be :obj:`torch.Tensor`): The tensors to place on :obj:`self.device`.
        Return:
            :obj:`Dict[str, torch.Tensor]`: The same as :obj:`inputs` but on the proper device.
        """
        return {
            name: tensor.to(self.device) if isinstance(tensor, torch.Tensor) else tensor
            for name, tensor in inputs.items()
        }