# This experiment file it is quite heavy on loading considering
# the imported packages
import datetime as dt
import enum
import json
import os
import sys
from pathlib import Path
from typing import Tuple

import hyperopt
import hyperopt.pyll
import numpy as np
from datasets import list_metrics
from transformers import logging as hf_logging

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from utils import get_project_path
from models.Model import ModelConstruction
from models.transformersModel import TransformersModel

PROJECT_DIRECTORY = get_project_path()

hf_logging.set_verbosity_error()
hf_logging.enable_explicit_format()


# Here are the possible model
# types denoted
class ModelType(enum.Enum):
    transformers = "transformers"
    bagOfWords2LayerModel = "BagOfWords2LayerModel"

class TokenizerType(enum.Enum):
    transformers = "transformers"


class ReportJSONEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return super(ReportJSONEncoder, self).default(obj)


def report(info: dict, reportPath: str):
    """ This function adds a report of an experiment to a json report file.
    The final reported experiment is presented in an html file in github.

    Args:
        reportPath (str): The json file to write or append the report to.
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
        fw.write(json.dumps(alreadyReported, indent=4, cls=ReportJSONEncoder))


def processTransformersLog(log_history: list) -> Tuple[dict, dict]:
    """
    This function preprocess the log history from transformers before reporting.
    The type of the log_history is always list. The items from 1 to N-1 are evaluation results.
    ... and the last item is a training summary.
    The keys in evaluation results are: eval_loss, eval_accuracy, eval_runtime, eval_samples_per_second, epoch, step
    ... and the keys in training summary are: train_runtime, train_samples_per_second, total_flos, epoch, step

    Args:
        log_history (list): The training log returned by transformers.trainer.state.log_history

    Returns:
        Tuple[dict, dict]: the last evaluation log, and the training summary
    """
    last_eval_state = log_history[-2]
    training_state = log_history[-1]
    return last_eval_state, training_state


def launchExperimentFromDict(d: dict, reportPath: str = None):
    """ This function launches experiment from a dictionary.
    The experiment configuration can use the hyperopt package or not.
    Also it can specify parameters regarding the model configuration, tokenizer configuration (tokenizer_config) and training configuration (args)
    Args:
        d (dict): [description]
        reportPath (str, optional): The json file to write or append the report. to. Defaults to './report.json'.
    """
    if reportPath is None:
        reportPath = os.path.join(PROJECT_DIRECTORY, 'src', 'experimentConfigs', 'report.json')
    model = ModelConstruction  # default model which does nothing

    # check if model type is of type transformers
    # if ModelType gets more than 3 types this should be changed
    # to a larger match case
    if d['model_type'] == ModelType.transformers.value:
        # TODO: transformers model is used, but a general model is needed here
        model_name_or_path = d['model_name_or_path']
        tokenizer_name_or_path = d.get('tokenizer_name_or_path', model_name_or_path)
        model = TransformersModel(modelName_or_pipeLine=model_name_or_path,
                                  tokenizer_name_or_path=tokenizer_name_or_path,
                                  fast_tokenizer=d.get('fast_tokenizer'),
                                  text_pre_cleaning=d.get('text_pre_cleaning', 'default'))

    if type(d['metric']) is str:
        d['metric'] = [d['metric']]
    assert (d['metric'][0] in list_metrics()), \
        f"The metric for evaluation is not supported.\n" \
        f"It should be in https://huggingface.co/metrics"

    model.registerMetric(*d['metric'])

    model.loadData(ratio=d['data_load_ratio'])

    hyperoptActive = d.get('use_hyperopt', False)
    if not hyperoptActive:
        _ = model.trainModel(
            train_val_split_iterator=d['args'].get('train_val_split_iterator', "train_test_split"),
            model_config=d['model_config'],
            tokenizer_config=d['tokenizer_config'],
            trainer_config=d['args'],
        )
        model.save()
        best_model_metric = model.getBestMetric()
        best_model_epoch = model.getBestModelEpoch()
        model_saved_path = model.training_saving_path
        report_description = d.pop('description')
        info_dict = {"description": report_description,
                     "results": {d['args']['metric_for_best_model']: best_model_metric},
                     "output_dir": str(model_saved_path),
                     "time_stamp": str(dt.datetime.now()),
                     "stopped_epoch": best_model_epoch,
                     **d}
        with open(Path(str(model_saved_path), 'report.json'), 'w') as fw:
            fw.write(json.dumps(info_dict, indent=4, cls=ReportJSONEncoder))
        report(info=info_dict,
               reportPath=reportPath)
    else:
        # if use_hyperopt = True inside the dictionary
        # prepare hyperopt to run for various values
        # search for values having this dictionary structure:
        # {"use_hyperopt" , "hyperopt_function", "arguments"}
        # see robertaHyperopt.json for more details.
        space = {argName: getHyperoptValue(argName, argValue)
                 for argName, argValue in d['args'].items()}

        all_evals = []

        def getEvals(args):
            # find which arguments use hyperopt
            # and stop them from being a dictionary
            actualArgs = {}

            for argName, argVal in args.items():
                if type(d['args'][argName]) is dict:
                    if d['args'][argName].get("use_hyperopt", False) and type(argVal) is dict:
                        actualArgs[argName] = argVal[argName]
                    else:
                        actualArgs[argName] = argVal
                else:
                    actualArgs[argName] = argVal

            # test the model
            # and get evaluations
            _ = model.trainModel(
                train_val_split_iterator=actualArgs.get('train_val_split_iterator', "train_test_split"),
                model_config=d['model_config'],
                tokenizer_config=d['tokenizer_config'],
                trainer_config=actualArgs)
            best_model_metric = model.getBestMetric()
            model_saved_path = model.getBestModelCheckpoint()
            all_evals.append({actualArgs['metric_for_best_model']: best_model_metric})
            return best_model_metric

        def getEvalsError(args):
            evals = getEvals(args)
            print(f"New evals = {evals}")
            # res = 100 - np.sum(evals) / np.size(evals)
            return evals

        bestHyperparametersDictFromHyperOpt = hyperopt.fmin(getEvalsError, space, hyperopt.tpe.suggest,
                                                            max_evals=d['hyperopt_max_evals'])
        bestHyperparametersDict = d['args'].copy()
        for k, v in bestHyperparametersDictFromHyperOpt.items():
            bestHyperparametersDict[k] = v
        report(info={**bestHyperparametersDict,
                     "all_evals": all_evals,
                     "results": float((np.max([v[d['args']['metric_for_best_model']] for v in all_evals]))),
                     "output_dir": f'./results/{model._modelName}',  # for server make this absolute server
                     "time_stamp": str(dt.datetime.now())},
               reportPath=reportPath)


def getHypervisorFunction(funcName: str) -> callable:
    """maps a function name from the hyperopt package to the function in the hyperopt package"""
    d = {
        "normal": hyperopt.hp.normal,
        "lognormal": hyperopt.hp.lognormal,
        "loguniform": hyperopt.hp.loguniform,
        "qlognormal": hyperopt.hp.qlognormal,
        "qnormal": hyperopt.hp.qnormal,
        "randint": hyperopt.hp.randint,
        "uniform": hyperopt.hp.uniform,
        "uniformint": hyperopt.hp.uniformint,
        "choice": hyperopt.hp.choice,
        "pchoice": hyperopt.hp.pchoice
    }
    assert (funcName in d.keys()), f"{funcName} not in supported hp functions"
    return d.get(funcName)


def getHyperoptValue(name: str, val: any):
    """This gets a value and if it is a dictionary it transforms it to a dictionary {name:value} where value is a function from the hyperopt package  
    Only use this if hyperopt_active is true on the first level of the dictionary configuration.
    """
    USE_HYPEROPT = "use_hyperopt"
    HYPEROPT_FUNC = "hyperopt_function"
    HYPEROPT_ARGS = "arguments"
    if type(val) is dict:
        answers = [k in val.keys() for k in [USE_HYPEROPT, HYPEROPT_FUNC, HYPEROPT_ARGS]]
        if np.all(answers):
            # actual hyperopt descriptor
            if val[USE_HYPEROPT]:
                hpfunc = getHypervisorFunction(val[HYPEROPT_FUNC])
                return {name: hpfunc(name, **val[HYPEROPT_ARGS])}
            else:
                print("Error: Having a hyperopt descriptor but hyperopt usage is not active")
                assert (False)
                return {name: ""}
        else:
            # a key-value has been forgotten
            assert not (np.any(answers))
            return val
    else:
        return val


def launchExperimentFromJson(fpath: str, reportPath: str):
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


def main(args: list):
    """ The main function of the program. It launches an experiment from a json file specified and reports
    to a file specified, else it reports to docs/report.json.
    use args:
    - test_path=<your test path> for setting the path of the test json file
    - report_path=<your report destination path> for setting the path for the report to be written or appended. 
    call it like:
    python experimentConfigs/experiment.py test_path=experimentConfigs/robertaDefault.json report_path=report.json
    Args:
        args (list): a dictionary containing the program arguments (sys.argv)
    """
    # Set the cache directory to /cluster/scratch if running on the cluster
    if Path().resolve().parts[1] == 'cluster':
        os.environ["TRANSFORMERS_CACHE"] = os.path.join(os.getenv("SCRATCH"), '.cache/huggingface/')

    argv = {a.split('=')[0]: a.split('=')[1] for a in args[1:]}
    testPath = argv.get('test_path', None)
    reportPath = argv.get('report_path', './report.json')
    if testPath is None:
        print("No test_path specified")
        exit(0)
    launchExperimentFromJson(testPath, reportPath)


if __name__ == "__main__":
    if Path().resolve().parts[1] == 'cluster':
        os.environ["WANDB_DISABLED"] = "true"  # for cluster we need to disable this
    main(sys.argv)
