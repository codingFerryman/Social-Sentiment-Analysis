# This experiment file it is quite heavy on loading considering
# the imported packages
import json
import hyperopt
import os, sys
import enum
import tensorflow as tf
import transformers
sys.path.append(os.path.join(os.path.dirname(__file__),'..'))
from models.Model import ModelConstruction
from models import transformersModel
from models import bagOfWords2LayersModel
from model.modelMaps import mapStrToTransformerModel, getModelMapAvailableNames
from preprocessing.pretrainedTransformersPipeline import PretrainedTransformersPipeLine
from preprocessing.pipelineMaps import mapStrToTransformersTokenizer, getTokenizerMapAvailableNames


# Here are the possible model
# types denoted
class ModelType(enum.Enum):
    transformers = "transformers"
    bagOfWords2LayerModel = "BagOfWords2LayerModel"

class TokenizerType(enum.Enum):
    transformers = "transformers"
    
def report():
    pass

def launchExperimentFromDict(d:dict):
    assert(d['model'] in getModelMapAvailableNames(), "No such model {}".format(d['model']))
    assert(d['tokenizer'] in getTokenizerMapAvailableNames), "No such tokenizer {}".format(d['tokenizer']))
    model = ModelConstruction # default model which does nothing

    # check if model type is of type transformers
    # if ModelType gets more than 3 types this should be changed
    # to a larger match case
    if (d['model_type'] == ModelType.transformers.value):
        name = d['model']
        model = transformersModel.TransformersModel(pipeLine={'modelName': name}, modelName=name, **d)
    # By default choose the sparse categorical accuracy
    model.registerMetric(tf.keras.metrics.SparseCategoricalAccuracy('accuracy'))
    for metric in d['metrics']:
        model.registerMetric(tf.keras.metrics.get(metric))
    
    if(d['tokenizer_type'] != TokenizerType.transformers.value):
        name = d['tokenizer']
        # TODO: transformers model is used, but no transformers model is needed here
        tokenizer = mapStrToTransformersTokenizer(name)
        model.pipeLine = transformersModel.getTransformersTokenizer(mapStrToTransformersTokenizer())

    model.testModel(**d['args'])

def launchExperimentFromJson(fpath: str):
    if not os.path.exists(fpath):
        raise Exception(f"No json fount at {fpath}")
    with open(fpath, 'r') as fr:
        experimentSettings = json.load(fr)
        launchExperimentFromDict(experimentSettings)



def main(args):
    pass


if __name__ == "__main__":
    main(sys.argv)