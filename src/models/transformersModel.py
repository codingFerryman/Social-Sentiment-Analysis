import tensorflow as tf
import transformers
from transformers import RobertaConfig
import numpy as np
import sklearn
from sklearn import model_selection
from sklearn.metrics import accuracy_score
import typing
import os, sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from preprocessing.pretrainedTransformersPipeline import PretrainedTransformersPipeLine, torchOrTFEnum
from models.Model import ModelConstruction
from modelMaps import mapStrToTransformerModel
from preprocessing.pipelineMaps import mapStrToTransformersTokenizer
import inputFunctions
import loggers
import pdb

logger = loggers.getLogger("RobertaModel", True)

def getDefaultTokenizer(loadFunction=None):
    if loadFunction == None:
        return PretrainedTransformersPipeLine(tokenizer=transformers.RobertaTokenizer, 
                                                        pretrainedTokenizerName='roberta-base')
    else:
        return PretrainedTransformersPipeLine(loadFunction=loadFunction, tokenizer=transformers.RobertaTokenizer, 
                                                        pretrainedTokenizerName='roberta-base')

def getTransformersTokenizer(transformersModelName:str, loadFunction:typing.Callable=None) -> PretrainedTransformersPipeLine:
    if loadFunction == None:
        return PretrainedTransformersPipeLine(tokenizer=mapStrToTransformersTokenizer(transformersModelName))
    else:
        return PretrainedTransformersPipeLine(loadFunction=loadFunction, tokenizer=mapStrToTransformersTokenizer(transformersModelName))

def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    acc = accuracy_score(labels, preds)
    return {
        'accuracy': acc,
    }



class TransformersModel(ModelConstruction):
    def __init__(self, dataPath:str=None, pipeLine=None, loadFunction=None, modelName:str="roberta"):
        self.configuration = transformers.RobertaConfig()
        if pipeLine == None:
            self.pipeLine = getDefaultTokenizer(loadFunction=loadFunction)
        elif type(pipeLine) == type({}):
            self.pipeLine = getTransformersTokenizer(pipeLine['modelName'], loadFunction)
        else:
            self.pipeLine = pipeLine
        self._registeredMetrics = []
        self._modelName = modelName

    def loadData(self):
        self.pipeLine.loadData()
        # self.pipeLine.trainTokenizer()
        self._dataLoaded = True
    
    def createModel(self, **kwargs) -> typing.Union[transformers.PreTrainedModel, tf.keras.Model]:
        assert self._dataLoaded, "data should be loaded before calling createModel"
        # assert self.pipeLine.num_words != None, "pipeline should have num_words != None"
        model = mapStrToTransformerModel(self._modelName)
        return model

    def testModel(self, train_val_split_iterator: typing.Iterator = [sklearn.model_selection.train_test_split], **kwargs):
        logger.info("Starting testing of RobertaModel")
        num_epochs = kwargs['epochs']
        for i, train_test_split in enumerate(train_val_split_iterator):
            logger.debug(f'{i}-th enumeration of train_val split iterator under cross validation')
            self.model = self.createModel()
            # optimizer = tf.keras.optimizers.Adam(learning_rate=3e-5, epsilon=1e-08, clipnorm=1.0)
            # loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
            if callable(getattr(self.model, 'compile', None)): # if tf model
                train_dataset, val_dataset = self.pipeLine.getEncodedDataset(train_test_split)
                # self.model.compile(optimizer=optimizer, loss=loss, metrics=self._registeredMetrics)
                # self.model.fit(train_dataset, epochs=num_epochs)
                training_args = transformers.TFTrainingArguments(
                    output_dir='./results',          # output directory
                    num_train_epochs=2,              # total # of training epochs
                    per_device_train_batch_size=64,  # batch size per device during training
                    per_device_eval_batch_size=64,   # batch size for evaluation
                    warmup_steps=10,                 # number of warmup steps for learning rate scheduler
                    weight_decay=0.01,               # strength of weight decay
                    logging_dir='./logs',            # directory for storing logs
                )
                trainer = transformers.TFTrainer(
                    model=self.model,                # the instantiated 🤗 Transformers model to be trained
                    args=training_args,              # training arguments, defined above
                    train_dataset=train_dataset,     # tensorflow_datasets training dataset
                    eval_dataset=val_dataset,        # tensorflow_datasets evaluation dataset
                    compute_metrics=compute_metrics  # metrics to compute while training
                )
            else:# if pytorch model
                train_dataset, val_dataset = self.pipeLine.getEncodedDataset(train_test_split, tfOrPyTorch=torchOrTFEnum.TORCH)
                training_args = transformers.TrainingArguments(
                    output_dir='./results',          # output directory
                    num_train_epochs=2,              # total number of training epochs
                    per_device_train_batch_size=64,  # batch size per device during training
                    per_device_eval_batch_size=64,   # batch size for evaluation
                    warmup_steps=10,                 # number of warmup steps for learning rate scheduler
                    weight_decay=0.01,               # strength of weight decay
                    logging_dir='./logs',            # directory for storing logs
                    logging_steps=10,
                )
                trainer = transformers.Trainer(
                    model=self.model,                # the instantiated 🤗 Transformers model to be trained
                    args=training_args,              # training arguments, defined above
                    train_dataset=train_dataset,     # training dataset
                    eval_dataset=val_dataset,        # evaluation dataset
                    compute_metrics=compute_metrics  # metrics to compute while training
                )
                trainer.train()

    def getTestResults(self) -> typing.List[dict]:
        """This method gets results from last training

        Returns:
            typing.List[dict]: list of dictionaries containing metric results
        """
        return self.currentResults

    def registerMetric(self, metric: 'tf.keras.metrics.Metric'):
        self._registeredMetrics.append(metric)

    def save(self, model_path: str, model_id: int):
        logger.info("Saving TransformersModel")
        self.model_params["class"] = self.__class__.__name__
        with open(os.path.join(model_path, 'params.json'), 'w') as json_file:
            json.dump(self.model_params, json_file)
        self.model.save(os.path.join(model_path, f'{self._modelName}_{model_id}.h5'))

    @staticmethod
    def load(load_folder_path:str, model_name:str, model_id:int):
        return BaseJointTransformerModel.load_model_by_class(JointTransRobertaModel, load_folder_path, f'{model_name}_{model_id}.h5')