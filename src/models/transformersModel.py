import os
import pathlib
import time
import typing
from pathlib import Path

import numpy as np
import transformers
from datasets import load_metric
from transformers import AutoModelForSequenceClassification, AutoConfig
from transformers import EarlyStoppingCallback
from transformers import TrainingArguments, Trainer

from models.Model import ModelConstruction, get_iterator_splitter_from_name
from preprocessing.cleaningText import cleaningMap
from preprocessing.pretrainedTransformersPipeline import PretrainedTransformersPipeLine
from utils import get_project_path, get_transformers_layers_num, loggers
from utils.diskArray import DiskArray

logger = loggers.getLogger("TransformersModel", True)


def getTransformersTokenizer(
        transformersModelName: str = None,
        loadFunction: typing.Callable[[str], typing.Tuple[list, list, list]] = None,
        **kwargs
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
        return PretrainedTransformersPipeLine(model_name_or_path=transformersModelName,
                                              fast_tokenizer=kwargs.get('fast_tokenizer'))
    else:
        return PretrainedTransformersPipeLine(model_name_or_path=transformersModelName,
                                              loadFunction=loadFunction,
                                              fast_tokenizer=kwargs.get('fast_tokenizer'))


class TransformersModel(ModelConstruction):
    def __init__(self, modelName_or_pipeLine=None, tokenizer_name_or_path=None,
                 loadFunction=None, fast_tokenizer=None,
                 text_pre_cleaning='default'):
        if modelName_or_pipeLine is None:
            # Set the default model to roberta-base
            modelName_or_pipeLine = "roberta-base"
        if tokenizer_name_or_path is None:
            tokenizer_name_or_path = modelName_or_pipeLine

        if type(modelName_or_pipeLine) is str:
            # If the value passed is a model's name
            self.pipeLine = getTransformersTokenizer(tokenizer_name_or_path, loadFunction,
                                                     fast_tokenizer=fast_tokenizer)
            self._modelName = modelName_or_pipeLine
            self._dataLoaded = False
        else:
            # If the value passed is a pipeline
            self.pipeLine = modelName_or_pipeLine
            self._modelName = modelName_or_pipeLine.tokenizer.name_or_path
            self._dataLoaded = self.pipeLine.is_data_loaded()

        self.text_pre_cleaning_function = cleaningMap(text_pre_cleaning)
        # Initialise some variables
        self._registeredMetrics = []
        self.metric = ('accuracy',)
        self.model = None
        self.trainer = None
        self.project_directory = get_project_path()

        # Training's logging path
        saving_relative_path = Path('trainings', self._modelName, time.strftime("%Y%m%d-%H%M%S"))

        self.training_saving_path = Path(self.project_directory, saving_relative_path)

        if pathlib.Path().resolve().parts[1] == 'cluster':
            self.training_saving_path_cluster = Path(os.getenv("SCRATCH"), 'cil-project', saving_relative_path)

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
        if _config.pad_token_id is None:
            logger.info('Set the pad_token_id to eos_token_id as there is no padding token in the config')
            _config.pad_token_id = _config.eos_token_id

        if pathlib.Path().resolve().parts[1] == 'cluster':
            if os.getenv("TRANSFORMERS_CACHE") is None:
                cache_dir = os.path.join(os.getenv("SCRATCH"), '.cache/huggingface/')
            else:
                cache_dir = os.getenv("TRANSFORMERS_CACHE")
            model = AutoModelForSequenceClassification.from_pretrained(self._modelName, config=_config,
                                                                       proxies={'http': 'proxy.ethz.ch:3128'},
                                                                       cache_dir=cache_dir)
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
        stratify = trainer_config.get('stratify', True) # if no stratify is specified, then assume it is true
        assert(not(stratify and "cross_validate_accuracy" == train_val_split_iterator)), f"stratify should be = false for {train_val_split_iterator}"
        splitter = get_iterator_splitter_from_name(train_val_split_iterator)

        # The callable function to pre-cleaning the texts
        encodedDatasetArgs = {'splitter': splitter,
                              'tokenizerConfig': tokenizer_config,
                              'cleaning_function': self.text_pre_cleaning_function}
        if 'test_size' in trainer_config.keys():
            encodedDatasetArgs['test_size'] = trainer_config['test_size']
        if 'stratify' in trainer_config.keys():
            encodedDatasetArgs['stratify'] = stratify

        trainer_state_log_history = DiskArray()
        trainers = DiskArray()
        allMetrics = []
        # get the dataset to work on
        for train_dataset, val_dataset in self.pipeLine.getEncodedDataset(**encodedDatasetArgs):
            logger.info("New train/val dataset pair for training")
            # Fine tune on last N layers
            if trainer_config:
                frozen_config = trainer_config['fine_tune_layers']
                self.model = self.createModel(model_config)
                if frozen_config['freeze']:
                    frozen_layers = self.get_frozen_layers(self.model,
                                                        frozen_config['num_unfrozen_layers'],
                                                        frozen_config['unfrozen_embeddings'])
                    for name, param in self.model.named_parameters():
                        for frozen_name in frozen_layers:
                            if frozen_name in name:
                                param.requires_grad = False
                trainer_config_copy = {**trainer_config.copy(), **kwargs}
                logger.info("Copying trainer_config")
            else:
                self.model = self.createModel()
                trainer_config_copy = {
                    "epochs": 1,
                    "batch_size": 128,
                    **kwargs,
                }
                logger.info("No trainer_config provided assuming minimum trainer_config_copy")

            # remove elements that are not required by transformers.Trainer
            for key in ["fine_tune_layers", "train_val_split_iterator", "stratify", "test_size"]:
                if key in trainer_config_copy.keys():
                    trainer_config_copy.pop(key)
            # Adapt configuration to Huggingface Trainer
            if "epochs" in trainer_config_copy.keys():
                logger.info("renaming epochs -> num_train_epochs")
                trainer_config_copy["num_train_epochs"] = trainer_config_copy.pop("epochs")
            if "batch_size" in trainer_config.keys():
                logger.info("renaming batch_size -> per_device_train_batch_size, batch_size --> per_device_eval_batch_size")
                trainer_config_copy["per_device_train_batch_size"] = trainer_config_copy.pop("batch_size")
                trainer_config_copy["per_device_eval_batch_size"] = trainer_config_copy["per_device_train_batch_size"]
            # Enable half precision training by default on the cluster
            if "fp16" not in trainer_config_copy.keys():
                if pathlib.Path().resolve().parts[1] == 'cluster':
                    logger.info("fp16 override to true for this cluster")
                    trainer_config_copy["fp16"] = True
            
            callbacks = []
            if "early_stopping_patience" in trainer_config_copy.keys():
                early_stopping_patience = trainer_config_copy.pop("early_stopping_patience")
                early_stopping_threshold = trainer_config_copy.pop("early_stopping_threshold", 0)
                callbacks.append(EarlyStoppingCallback(early_stopping_patience=early_stopping_patience,
                                                       early_stopping_threshold=early_stopping_threshold))

            if pathlib.Path().resolve().parts[1] == 'cluster':
                training_logging_dir = self.training_saving_path_cluster
            else:
                training_logging_dir = self.training_saving_path

            training_args = TrainingArguments(
                # Please do NOT add logging_dir here (that is for TensorBoard)
                # Please save checkpoints to scratch when training on the cluster
                output_dir=training_logging_dir,
                **trainer_config_copy,
                save_strategy='epoch'
            )
            logger.debug(f"The program is running from: {str(pathlib.Path().resolve())}")
            logger.debug(f"The checkpoints will be saved in {training_logging_dir}")
            assert Path(training_args.output_dir) == Path(training_logging_dir), \
                "The logging directory for checkpoints is not correct"
            self.trainer = Trainer(
                model=self.model,
                args=training_args,
                train_dataset=train_dataset,
                eval_dataset=val_dataset,
                compute_metrics=self.compute_metrics,
                callbacks=callbacks,
            )
            self.trainer.train()
            trainers.append([self.trainer.state, self.trainer.model])
            trainer_state_log_history.append(self.trainer.state.log_history)
            allMetrics.append(self.trainer.state.best_metric)

        # get best trainer strategy
        logHistory, self.trainer, self.bestMetric = self.getBestTrainerStrategy(allMetrics, trainers, trainer_state_log_history, train_dataset , val_dataset)

        return logHistory

    def getBestTrainerStrategy(self, allMetrics: list, trainerList: DiskArray, trainer_state_log_history: DiskArray, train_dataset , eval_dataset):
        logger.info(f"get best trainer strategy for len(allMetrics) = {len(allMetrics)}")
        bestMetricIndex = int(np.argmax(allMetrics))
        state, model = trainerList[bestMetricIndex]
        t = Trainer(model=model,
                    compute_metrics=self.compute_metrics,
                    train_dataset=train_dataset,
                    eval_dataset=eval_dataset)
        t.state = state
        logger.info("{}{}".format(t.state.best_metric, np.max(allMetrics)))
        return trainer_state_log_history[bestMetricIndex], t, float(np.average(allMetrics))

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
        logger.debug("The metrics in this eval: {}".format(str(result)))
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

    def getBestModelEpoch(self):
        return self.trainer.state.epoch

    def getTokenizer(self):
        return self.pipeLine.getTokenizer()

    def save(self, model_path: str = None):
        if model_path is None:
            model_path = Path(self.training_saving_path, 'model')
        logger.info("Saving TransformersModel")
        self.trainer.save_state()
        self.trainer.save_model(model_path)
        self.trainer.state.save_to_json(Path(self.training_saving_path, 'trainer_state.json'))
        _tokenizer = self.getTokenizer()
        _tokenizer.save_pretrained(Path(self.training_saving_path, 'tokenizer'))
