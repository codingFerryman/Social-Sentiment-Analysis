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
from models.modelMaps import mapStrToTransformerModel, getModelMapAvailableNames
from preprocessing.pretrainedTransformersPipeline import PretrainedTransformersPipeLine
from preprocessing.pipelineMaps import mapStrToTransformersTokenizer, getTokenizerMapAvailableNames


# Here are the possible model
# types denoted
class ModelType(enum.Enum):
    transformers = "transformers"
    bagOfWords2LayerModel = "BagOfWords2LayerModel"

class TokenizerType(enum.Enum):
    transformers = "transformers"
    
def report(info:dict, reportPath:str):
    """ This function adds a report of an experiment to a json report file.
    The final reported experiment is presented in an html file in github.

    Args:
        reportPath (str): The json file to write or append the report. to.
    """
    if not os.path.exists(reportPath):
        alreadyReported = {'num_experiments': 0, 
                           'experiments': []}
    else:
        with open(reportPath, 'r') as fr:
            alreadyReported = json.load(fr)
    alreadyReported['num_experiments'] = alreadyReported['num_experiments'] + 1
    experiments = alreadyReported['experiments'] 
    experiments.append(info)
    alreadyReported['experiments'] = experiments
    with open(reportPath, 'w') as fw:
        json.dump(alreadyReported, fw)

def launchExperimentFromDict(d:dict, reportPath:str='./report.json'):
    """ This function launches experiment from a dictionary.

    Args:
        d (dict): [description]
        reportPath (str, optional): The json file to write or append the report. to. Defaults to './report.json'.
    """
    assert(d['model'] in getModelMapAvailableNames(), "No such model {}".format(d['model']))
    assert(d['tokenizer'] in getTokenizerMapAvailableNames(), "No such tokenizer {}".format(d['tokenizer']))
    model = ModelConstruction # default model which does nothing

    # check if model type is of type transformers
    # if ModelType gets more than 3 types this should be changed
    # to a larger match case
    if (d['model_type'] == ModelType.transformers.value):
        name = d['model']
        model = transformersModel.TransformersModel(pipeLine={'modelName': name}, modelName=name, **d)
    # By default choose the sparse categorical accuracy
    # model.registerMetric(tf.keras.metrics.SparseCategoricalAccuracy('accuracy'))
    model.registerMetric({'name': 'accuracy'})
    for metric in d.get('metrics',[]):
        # model.registerMetric(tf.keras.metrics.get(metric))
        model.registerMetric({'name': metric})
    
    if(d['tokenizer_type'] != TokenizerType.transformers.value):
        name = d['tokenizer']
        # TODO: transformers model is used, but a general model is needed here
        tokenizer = mapStrToTransformersTokenizer(name)
        model.pipeLine = transformersModel.getTransformersTokenizer(mapStrToTransformersTokenizer())
    model.loadData()
    evals = model.testModel(**d['args'])
    report(info={**d, 
            "results": evals,
            "output_dir": f'./results/{model._modelName}'}, # for server make this absolute server
           reportPath=reportPath)

def launchExperimentFromJson(fpath:str, reportPath:str):
    """This launches experiment described in a json file.
    It reads a json file it transforms is to  dict and calls the launchExperimentFromDict
    function.

    Args:
        fpath (str): The path of the json file

    Raises:
        FileNotFoundError: No json found at path if path does not exist
    """
    if not os.path.exists(fpath):
        raise FileNotFoundError(f"No json fount at {fpath}")
    with open(fpath, 'r') as fr:
        experimentSettings = json.load(fr)
        launchExperimentFromDict(experimentSettings, reportPath)



def main(args:list):
    """ The main function of the program. It launches an experiment from a json file specified and reports
    to a file specified, else it reports to ./report.json.
    use args:
    - test_path=<your test path> for setting the path of the test json file
    - report_path=<your report destination path> for setting the path for the report to be written or appended. 
    call it like:
    python experimentConfigs/experiment.py test_path=experimentConfigs/robertaDefault.json report_path=report.json
    Args:
        args (list): a dictionary containing the program arguments (sys.argv)
    """
    argv = {a.split('=')[0]:a.split('=')[1] for a in args[1:]}
    testPath = argv.get('test_path', None)
    reportPath = argv.get('report_path', './report.json')
    if testPath is None:
        print("No test_path specified")
        exit(0)
    launchExperimentFromJson(testPath, reportPath)


if __name__ == "__main__":
    main(sys.argv)