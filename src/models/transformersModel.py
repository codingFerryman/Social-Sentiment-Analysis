import time
import typing
import pathlib
from pathlib import Path

import numpy as np
import transformers
from datasets import load_metric
from transformers import AutoModelForSequenceClassification, AutoConfig
from transformers import EarlyStoppingCallback
from transformers import TrainingArguments, Trainer

from models.Model import ModelConstruction, get_iterator_splitter_from_name
from preprocessing.pretrainedTransformersPipeline import PretrainedTransformersPipeLine
from utils import get_project_path, get_transformers_layers_num, loggers

logger = loggers.getLogger("TransformersModel", True)


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

class TransformersModel(ModelConstruction):
    def __init__(self, modelName_or_pipeLine=None, loadFunction=None):
        self.configuration = transformers.RobertaConfig()
        if modelName_or_pipeLine is None:
            # Set the default model to roberta-base
            modelName_or_pipeLine = "roberta-base"
        if type(modelName_or_pipeLine) is str:
            # If the value passed is a model's name
            self.pipeLine = getTransformersTokenizer(modelName_or_pipeLine, loadFunction)
            self._modelName = modelName_or_pipeLine
            self._dataLoaded = False
        else:
            # If the value passed is a pipeline
            self.pipeLine = modelName_or_pipeLine
            self._modelName = modelName_or_pipeLine.tokenizer.name_or_path
            self._dataLoaded = self.pipeLine.is_data_loaded()

        # Initialise some variables
        self._registeredMetrics = []
        self.metric = ('accuracy',)
        self.model = None
        self.trainer = None
        self.project_directory = get_project_path()

        # Training's logging path
        self.training_saving_path = Path(self.project_directory, 'trainings', 'logging',
                                         self._modelName, time.strftime("%Y%m%d-%H%M%S"))

    def loadData(self, ratio='sub'):
        self.pipeLine.loadData(ratio)
        self._dataLoaded = True

    @staticmethod
    def get_frozen_layers(model, unfreeze_last_n_layers: int = 1, unfreeze_embeddings=False):
        """
        Get the keywords of the frozen layers
        Args:
            model: PyTorch model
            unfreeze_last_n_layers (int): Number of unfrozen layers (from bottom to top)
            unfreeze_embeddings (bool): freeze

        Returns:
            list: Keywords of frozen layers
        """
        if unfreeze_embeddings:
            frozen_layers = []
        else:
            frozen_layers = ['embeddings']

        total_layers = get_transformers_layers_num(model)
        last_frozen_layer = total_layers - unfreeze_last_n_layers

        frozen_layers += ['layer.' + str(i) for i in range(1, last_frozen_layer + 1)]
        return frozen_layers

    def createModel(self, model_config_dict: dict = None) -> typing.Union[transformers.PreTrainedModel]:
        assert self._dataLoaded, "data should be loaded before calling createModel"

        _config = AutoConfig.from_pretrained(self._modelName)
        if model_config_dict:
            _config.update(model_config_dict)
        if pathlib.Path().resolve().parts[1] == 'cluster':
            model = AutoModelForSequenceClassification.from_pretrained(self._modelName, config=_config,
                                                                       proxies={'http': 'proxy.ethz.ch:3128'})
        else:
            model = AutoModelForSequenceClassification.from_pretrained(self._modelName, config=_config)
        return model

    def trainModel(self, train_val_split_iterator: str = "train_test_split",
                   model_config: dict = None,
                   tokenizer_config: dict = None,
                   trainer_config: dict = None,
                   **kwargs):
        logger.debug(f"The split iterator is: {train_val_split_iterator}")
        logger.debug(f"The kwards: {kwargs}")
        logger.info(f"Starting testing of {self._modelName}")
        splitter = get_iterator_splitter_from_name(train_val_split_iterator)

        train_dataset, val_dataset = self.pipeLine.getEncodedDataset(splitter=splitter,
                                                                     tokenizerConfig=tokenizer_config,
                                                                     test_size=0.2)

        # Fine tune on last N layers
        if trainer_config:
            frozen_config = trainer_config.pop('fine_tune_layers')
            self.model = self.createModel(model_config)
            if frozen_config['freeze']:
                frozen_layers = self.get_frozen_layers(self.model,
                                                       frozen_config['num_unfrozen_layers'],
                                                       frozen_config['unfrozen_embeddings'])
                for name, param in self.model.named_parameters():
                    for frozen_name in frozen_layers:
                        if frozen_name in name:
                            param.requires_grad = False
        else:
            self.model = self.createModel()
            trainer_config = {
                "epochs": 1,
                "batch_size": 128
            }

        for k in kwargs.keys():
            trainer_config[k] = kwargs[k]
        # Adapt configuration to Huggingface Trainer
        if "epochs" in trainer_config.keys():
            trainer_config["num_train_epochs"] = trainer_config.pop("epochs")
        if "batch_size" in trainer_config.keys():
            trainer_config["per_device_train_batch_size"] = trainer_config.pop("batch_size")
            trainer_config["per_device_eval_batch_size"] = trainer_config["per_device_train_batch_size"]
        # Enable half precision training by default on the cluster
        if "fp16" not in trainer_config.keys():
            if pathlib.Path().resolve().parts[1] == 'cluster':
                trainer_config["fp16"] = True

        callbacks = []
        if "early_stopping_patience" in trainer_config.keys():
            early_stopping_patience = trainer_config.pop("early_stopping_patience")
            callbacks.append(EarlyStoppingCallback(early_stopping_patience=early_stopping_patience))

        training_args = TrainingArguments(
            logging_dir=Path(self.training_saving_path, 'logs'),
            output_dir=Path(self.training_saving_path, 'checkpoints'),
            **trainer_config
        )

        self.trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            compute_metrics=self.compute_metrics,
            callbacks=callbacks,
        )
        self.trainer.train()

        return self.trainer.state.log_history

    def registerMetric(self, *metric):
        """Register a metric for evaluation"""
        if type(metric) is str:
            metric = (metric,)
        self.metric = metric

    def compute_metrics(self, eval_pred=None) -> dict:
        """This function is used by TFtrainer and Trainer classes in the transformers library.
        However it can more bradly used to compute metrics during training or testing for evaluation.
        Args:
            eval_pred (object): An object containing label_ids (the groundtruth labels) and predictions (the logit predictions of the model)
        Returns:
            Evaluation results
        """
        metric = load_metric(*self.metric)
        logits, labels = eval_pred
        predictions = np.argmax(logits, axis=-1)
        result = metric.compute(predictions=predictions, references=labels)
        return result

    def getPipeLine(self):
        return self.pipeLine

    def getTrainer(self):
        return self.trainer

    def getLastEval(self):
        return self.trainer.state.log_history[-2]

    def getBestMetric(self):
        return self.trainer.state.best_metric

    def getBestModelCheckpoint(self):
        return self.trainer.state.best_model_checkpoint

    def save(self, model_path: str = None):
        if model_path is None:
            model_path = Path(self.training_saving_path, 'model')
        logger.info("Saving TransformersModel")
        self.trainer.save(model_path)
