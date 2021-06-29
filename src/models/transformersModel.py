import json
import os
import sys
import time
import typing
from pathlib import Path

import numpy as np
import transformers
from datasets import load_metric

from utilities import get_project_path

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from preprocessing.pretrainedTransformersPipeline import PretrainedTransformersPipeLine
from models.Model import ModelConstruction, get_iterator_splitter_from_name
# from preprocessing.pipelineMaps import mapStrToTransformersTokenizer
from transformers import AutoModelForSequenceClassification, AutoConfig
from transformers import TrainingArguments, Trainer
from transformers import EarlyStoppingCallback
import loggers
from icecream import ic

logger = loggers.getLogger("RobertaModel", True)


def getTransformersTokenizer(
        transformersModelName: str = None,
        loadFunction: typing.Callable[[str], typing.Tuple[list, list, list]] = None

) -> PretrainedTransformersPipeLine:
    """
    This function returns the transformers tokenizer respective with the transformers model name.
    Each transformers model uses a respective tokenizer with the same name.
    The loadFunction loads the dataset into a tuple with 3 lists: train_positive_tweets, train_negative_tweets, test_tweets.

    Args:
        transformersModelName (str):
            The name of the transformers model
        loadFunction (typing.Callable[[str], typing.Tuple[list,list,list]], optional):
            A callable load function that loads the dataset. Defaults to None.

    Returns:
        PretrainedTransformersPipeLine: The transformers pipeline with the pretrained tokenizer for the respective model.
            The tokenizer may be trained on a much different dataset than tweets
    """
    # If no model name is specified, default to 'bert-base-uncased'
    if transformersModelName is None:
        transformersModelName = 'bert-base-uncased'

    if loadFunction is None:
        return PretrainedTransformersPipeLine(model_name_or_path=transformersModelName)
    else:
        return PretrainedTransformersPipeLine(loadFunction=loadFunction,
                                              model_name_or_path=transformersModelName)


def compute_metrics(eval_pred, *args):
    """This function is used by TFtrainer and Trainer classes in the transformers library.
    However it can more bradly used to compute metrics during training or testing for evaluation.
    Args:
        eval_pred (object): An object containing label_ids (the groundtruth labels) and predictions (the logit predictions of the model)
        args (str): The strategy for metrics computation. Accuracy by default
    Returns:
        TODO: Complete the documentation
    """
    metric = load_metric(args or "accuracy")
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)


class TransformersModel(ModelConstruction):
    def __init__(self, modelName_or_pipeLine=None, loadFunction=None):
        self.configuration = transformers.RobertaConfig()
        if modelName_or_pipeLine is None:
            modelName_or_pipeLine = "roberta-base"
        if type(modelName_or_pipeLine) is str:
            self.pipeLine = getTransformersTokenizer(modelName_or_pipeLine, loadFunction)
            self._modelName = modelName_or_pipeLine
            self._dataLoaded = False
        else:
            self.pipeLine = modelName_or_pipeLine
            self._modelName = modelName_or_pipeLine.tokenizer.name_or_path
            self._dataLoaded = self.pipeLine.is_data_loaded()
        self._registeredMetrics = []

        self.model = None
        self.trainer = None
        self.project_directory = get_project_path()
        self.training_saving_path = Path(self.project_directory, 'trainings', 'logging',
                                         self._modelName, time.strftime("%Y%m%d-%H%M%S"))

    def loadData(self, ratio='sub'):
        self.pipeLine.loadData(ratio)
        self._dataLoaded = True

    def loadTokenizerConfig(self, config_path: str = None) -> typing.Dict:
        if config_path is None:
            config_path = Path(self.project_directory, 'src', 'experimentConfigs',
                               'transformersTokenizers',
                               'default.json')
        with open(config_path, 'r') as fc:
            return json.load(fc)

    def loadModelConfig(self, config_path: str = None) -> typing.Dict:
        if config_path is None:
            config_path = Path(self.project_directory, 'src', 'experimentConfigs',
                               'transformersModels',
                               'default.json')
        with open(config_path, 'r') as fm:
            return json.load(fm)

    def loadTrainerConfig(self, config_path: str = None) -> typing.Dict:
        if config_path is None:
            config_path = Path(self.project_directory, 'src', 'experimentConfigs',
                               'transformersTrainers',
                               'default.json')
        with open(config_path, 'r') as ft:
            return json.load(ft)

    @staticmethod
    def get_frozen_layers(model_name):
        # TODO: make it more reasonable
        if 'roberta-base' in model_name:
            num_layers = 12
            frozen_layers = ['embeddings'] + ['layer.' + str(i) for i in range(int(num_layers * 0.75))]
        elif 'roberta-large' in model_name:
            num_layers = 24
            frozen_layers = ['embeddings'] + ['layer.' + str(i) for i in range(int(num_layers * 0.75))]
        else:
            frozen_layers = ['embeddings'] + ['layer.']  # Train classifier only
        return frozen_layers

    def createModel(self, model_config_name_or_path: str = None,
                    filename_if_save: str = None
                    ) -> typing.Union[transformers.PreTrainedModel]:
        assert self._dataLoaded, "data should be loaded before calling createModel"
        _project_path = get_project_path()
        if model_config_name_or_path is None:
            model_config_name_or_path = self._modelName
        if not os.path.isfile(model_config_name_or_path):
            _config = AutoConfig.from_pretrained(model_config_name_or_path)
            config_dict = self.loadModelConfig()
            _config.update(config_dict)
        else:
            config_dict = self.loadModelConfig(model_config_name_or_path)
            _config = AutoConfig.from_pretrained(config_dict)
        if filename_if_save:
            if filename_if_save[-5:] != '.json':
                filename_if_save += '.json'
            _config_saving_path = Path(_project_path, 'src', 'experimentConfigs',
                                       'transformersModels',
                                       filename_if_save)
            _config.to_json_file(_config_saving_path, use_diff=False)
        model = AutoModelForSequenceClassification.from_config(_config)
        return model

    def trainModel(self, train_val_split_iterator: str = "train_test_split",
                   model_config_name_or_path=None,
                   tokenizer_config_dict_or_path=None,
                   trainer_config_dict_or_path=None,
                   freeze_model=False,
                   **kwargs):
        ic(train_val_split_iterator, kwargs)
        logger.info(f"Starting testing of {self._modelName}")
        # evals = []
        splitter = get_iterator_splitter_from_name(train_val_split_iterator)

        if type(tokenizer_config_dict_or_path) is not dict:
            tokenizer_config = self.loadTokenizerConfig(tokenizer_config_dict_or_path)
        else:
            tokenizer_config = tokenizer_config_dict_or_path

        train_dataset, val_dataset = self.pipeLine.getEncodedDataset(splitter=splitter,
                                                                     tokenizerConfig=tokenizer_config,
                                                                     test_size=0.2)
        self.model = self.createModel(model_config_name_or_path)
        if freeze_model:
            frozen_layers = self.get_frozen_layers(self._modelName)
            for name, param in self.model.named_parameters():
                for frozen_name in frozen_layers:
                    if frozen_name in name:
                        param.requires_grad = False
        logger.debug("training pytorch model")

        if type(trainer_config_dict_or_path) is not dict:
            trainer_config = self.loadTrainerConfig(trainer_config_dict_or_path)
        else:
            trainer_config = trainer_config_dict_or_path
        if "epochs" in kwargs.keys():
            trainer_config["num_train_epochs"] = kwargs["epochs"]
        if "batch_size" in kwargs.keys():
            trainer_config["per_device_train_batch_size"] = kwargs["batch_size"]
            trainer_config["per_device_eval_batch_size"] = kwargs["batch_size"]
        training_args = TrainingArguments(
            logging_dir=Path(self.training_saving_path, 'logs'),
            output_dir=Path(self.training_saving_path, 'checkpoints'),
            **trainer_config
        )

        early_stopping_patience = kwargs.pop('early_stopping_patience', 3)
        self.trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            compute_metrics=compute_metrics,
            callbacks=[EarlyStoppingCallback(early_stopping_patience=early_stopping_patience)],
        )
        self.trainer.train()

        return self.trainer.state.log_history

    def getTrainer(self):
        return self.trainer

    def getLastEval(self):
        return self.trainer.state.log_history[-2]

    def save(self, model_path: str = None):
        if model_path is None:
            model_path = Path(self.training_saving_path, 'model')
        logger.info("Saving TransformersModel")
        self.trainer.save(model_path)
