# [Computational Intelligence Lab Project 2](https://github.com/supernlogn/Computational-Intelligence-Lab)
This is our project ETHZ CIL Text Classification 2021 of CIL course in ETH Zurich.

## Contents
- [Results](#results)
- [Setup](#setup)
- [Execution](#execution)
    - [Experimentation on Leonhard](#experimentation-on-leonhard)
    - [Experimentation locally](#experimentation-locally)
    - [Submission to Kaggle](#submission to-kaggle)
    - [Cleaning the dataset](#cleaning the-dataset)
    - [See the impact of hashtags](#see-the-impact-of-hashtags)
    - [Majority voting](#majority-voting)
- [Configurations](#configurations)
    - [args](#args)
    - [tokenizer_config](#tokenizer-config)
    - [model_config](#model_config)
    - [Hyperparameter optimization](#hyperparameter-optimization)
- [Reproducing The Report's Work](#reproducing-the-report's-work)
    - [Create Table I results](#create-table-i-results)
    - [Create Table II results](#create-table-ii-results)
    - [Create Table III results](#create-table-iii-results)
    - [Create Table IV results](#create-table-iv-results)
    - [Create Table VI results](#create-table-vi-results)
    - [Create Table VII results](#create-table-vii-results)
    - [Majority voting results](#majority-voting-results)
- [Code Formatting](#code-formatting)
- [Repo structure](#repo-structure)
- [Dependencies](#dendencies)
- [Cluster training requirements](#cluster-training-requirements)
- [Developers - Students](#Developers---Students)
- [Acknowledgements](#acknowledgements)


## Results

- Results on validation data can be seen [here](https://supernlogn.github.io/Computational-Intelligence-Lab/)
- Results on test data can be seen in the respective Kaggle competition.

## Setup
To work with this project a certain procedure is needed. First clone this repo. Then we recommend using the automated scripts for setup. If the files where not cloned then a git init is executed without a specified origin, for some scripts to be able to be executed. These scripts create a virtual environment and download any data needed for this repo. Please be inside the ETHZ network while using them, to download the dataset.

Using an automated script:
```bash
cd <path-to-Computational-Intelligence-Lab-directory>
# gpu local
bash setup_local.sh
# leonhard cluster
bash setup_leonhard.sh
# Windows machine
".\setup_local_windows.ps1"
```

or doing it manually:
```bash
cd <path-to-Computational-Intelligence-Lab-directory>
if [ ! -d ./.git/ ]
then
git init
fi
mkdir -pv trainings # where training checkpoints are stored
python -m venv ./cil-venv
source venv/bin/activate # on windows this path differs
pip install -r requirements.txt
spacy download en_core_web_sm
# download data to the data folder
# start developing / running
```

❗ Note: For leonhard please make sure that the proxy http://proxy.ethz.ch:3128 has been set for downloading external models.

❗ Note: If you use a cpu and not a gpu, use [requirements_cpu.txt](./requirements_cpu.txt) rather than requirements.txt .

❗ Note: If you use a virtual machine from [lambdaLabs](https://lambdalabs.com/) use [requirements_lambdalabs.txt](./requirements_lambdalabs.txt) rather than requirements.txt

## Execution
There are many scripts and ways of execution that were tried during the development of this project. Here we list all of them, to execute them you need to be in their directory. Below we also include examples of running them. [ [Skip to examples](https://github.com/supernlogn/Computational-Intelligence-Lab#experimentation-on-leonhard) ]


- >bash [runExperimentOnLeonhard.sh](./src/experimentConfigs/runExperimentOnLeonhard.sh) `<leonhard-username> <path-to-configuration-inside-Leonhard>` This script submits a training of a model to run by the leonhard cluster in under 24 hours using a gpu. The model and the preprocessing are described by the configuration file provided. After running the training the validation results are stored inside the [report.json](./docs/report.json) and uploaded, so that they can be viewed in web.
- >bash [runExperimentAndUploadReportCluster.sh](./src/experimentConfigs/runExperimentAndUploadReportCluster.sh) `<path-to-configuration-inside-Leonhard>`  This script is executed inside the cluster. Its purpose is to load the modules and environment needed by the scripts to run, execute the training and then upload the report of the training. This should be executed inside the cluster using the bsub command. To submit this while inside the cluster you can type:
- >bash [runExperimentAndUploadReport.sh](./src/experimentConfigs/runExperimentAndUploadReport.sh) `<path-to-configuration-inside-your-pc>` This script loads the environment needed to execute the experiment.py and then executes the training of the model. After the training ends, the results are stored inside docs/report.json and uploaded to Github This script also assumes that a cil-venv environment exists in the base of the repo and tries to load it.
- >bash [uploadNewReport.sh](./src/experimentConfigs/uploadNewReport.sh) `<report-file-to-upload =../../docs/report.json>`  This scripts uploads the report.json to Github by committing it and pushing to the branch that the local Github project is currently on. It accepts one argument the path of the report file. The argument by default is `../../docs/report.json`.
- >`python` [experiment.py](./src/experimentConfigs/experiment.py) `test_path=<path-to-configuration> report_path=<path-to-the-report.json-file>` This script provides an entry to the framework and launches an experiment(training) from a configuration file. The configuration file is a json file containing the configuration for the model and the preprocessing. The validation results of the training are stored inside the json file specified. If this is not specified `docs/report.json` is used by default. Args for this script are the `test_path` for setting the path of the configuration json file and the `report_path` for setting the path for the report json file to be written or appended. The training's checkpoints will be stored inside the `<path-to-repo>/trainings` directory.
- >`python` [submission.py](./src/experimentConfigs/submission.py) `load_path=<directory-containing-'model'-and-'tokenizer'>  batch_size=<batch-size-used-for-the-model=128>  device=<gpu-id|cpu-id =cuda:0|cpu> text_path=<path-where-test_data.txt is =../../data/test_data.txt>` This script creates a submission csv file to be submitted to the kaggle competition. It loads an already trained model from its checkpoints. The device and the model's batch size can be also specified when calling it.
- >`python` [cleaningText.py](./src/preprocessing/cleaningText.py) `data_path=<path-text-file-with-a-tweet-per-line> output=<path-to-output-cleaned-text-file>` This script cleans/preprocess the text from a file and exports the cleaned data to another (new) file.
- >`python` [hastagExperiment.py](./src/experimentConfigs/hastagExperiment.py) `dataset=<use-full-or-partial-dataset> load_path=<trained-model-path> freq=500 prob=0.7` This script extracts hashtags and finds how much a hashtag matters. It is used to produce the hashtag analysis in the project's final report.
- > [clusterFeatures.ipynb](./src/explorations/clusterFeatures.ipynb) This notebook together with [createWordVecsFromModel.py](./src/explorations/createWordVecsFromModel.py) are used to create word vectors (output of last layers) based on a model and view them after performing PCA on them.

### Experimentation on Leonhard
Run these only after completing the setup for leonhard section.

From your pc's console:
```bash
cd <path-to-Computational-Intelligence-Lab-directory>
cd src/experimentConfigs
bash runExperimentOnLeonhard.sh <leonhard-username> <path-to-configuration-inside-Leonhard>
```

Example paths to the configuration:
```
/cluster/home/<leonhard-username>/Computational-Intelligence-Lab/src/configs/bertweet.json
/cluster/home/<leonhard-username>/Computational-Intelligence-Lab/src/configs/xlnet_base.json
/cluster/home/<leonhard-username>/Computational-Intelligence-Lab/src/configs/roberta_base.json
```

From inside leonhard:
```bash
module load gcc/6.3.0 python_gpu/3.8.5 hdf5/1.10.1 eth_proxy
cd <path-to-Computational-Intelligence-Lab-directory>
cd src/experimentConfigs
bsub -W 23:30 -n 8 -R "rusage[mem=9000,scratch=10000,ngpus_excl_p=1]" -R "select[gpu_mtotal0>=10000] \
./runExperimentAndUploadReportCluster.sh <path-to-configuration-inside-Leonhard>
```

### Experimentation locally
Run these only after completing the local setup section.

```bash
cd <path-to-Computational-Intelligence-Lab-directory>
source cil-venv/bin/activate
cd src/experimentConfigs
bash runExperimentAndUploadReport.sh <path-to-configuration-locally>
```
Typical paths to configuration locally are the paths to json files inside src/configs. 
e.g.

```
../bertweet.json
../xlnet_base.json
../configs/roberta_base.json
```

### Submission to Kaggle

Below is a command example for creating a valid kaggle submission. After this, a csv file is produced which can be submitted to Kaggle.

```bash
cd <path-to-Computational-Intelligence-Lab-directory>
source cil-venv/bin/activate
cd src/experimentConfigs
python submission.py load_path=../../trainings/roberta-base/20210709-102233 batch_size=128 \
text_path=../../data/test_data.txt
```

### Cleaning the dataset
Cleaning the dataset is a step that we tried in our data pipeline and is reported inside the final report. To start the cleaning procedure use the lines below:

```bash
cd <path-to-Computational-Intelligence-Lab-directory>
source cil-venv/bin/activate
cd src/preprocessing
python cleaningText.py data_path=../../data/train_pos_full.txt output=cleaned_data_pos_full.txt
python cleaningText.py data_path=../../data/train_neg_full.txt output=cleaned_data_neg_full.txt
# optionally we can overwrite previous data with the cleaned using the 2 lines below
# mv cleaned_data_pos_full.txt ../../data/train_pos_full.txt
# mv cleaned_data_neg_full.txt ../../data/train_neg_full.txt
```

### See the impact of hashtags

```bash
python hashtagExperiment.py dataset=full load_path=../../trainings/roberta-base/20210709-102233 freq=500 prob=0.7
```

### Majority voting
After running some models and having the submission files, a majority voting algorithm can be used. To use this see our majority voting [notebook](src/models/majority_voting.ipynb).



## Configurations
The configuration for each training is stored inside a json file. See [example](./src/configs/roberta_base.json). This json file specifies the parameters for preprocessing, tokenization and the model. 

The main parameters for the experiment/training to be launched are described by:

- >description: description of the experiment. This is later used as the experiment name inside the report.json,
- >model_name_or_path: The model name according to the huggingface transformers or a path to another model,
- >data_load_ratio: a number in the interval (0,1] indicating how many data will be used from the dataset,
- >model_type: type of model to use. It can be "transformers" for using the transformers model or other (left for future development) for using a model based on a bag of words. 
- >tokenizer_type: It can be "transformers" for using a transformers pretrained tokenizer,
- >fast_tokenizer: Whether to use a fast tokenizer implementation,

The 4 main parts of the configuration are then:
- >args : arguments of the training.
- >tokenizer_config : tokenizer's and preprocessing's config.
- >model_config : arguments for the model input-output.
- >metric : metrics to be evaluated on the validation data after each epoch.

### args

- > epochs: Number of training epochs.
- > batch_size: Batch size to use while training.
- > adafactor: Whether to use Ada rather than Adam for backpropagation.
- > warmup_steps: Number of warmup steps before starting the training.
- > weight_decay: Weight decay at each step.
- > learning_rate: Training's initial learning rate. It drops every epoch while training.
- > evaluation_strategy: "epoch" for per epoch evaluation or "steps" for per 500 steps evaluation.
- > logging_strategy: "epoch" for per epoch logging of performance or "steps" for per 500 steps logging of performance.
- > overwrite_output_dir: Whether to overwrite the output directory or to append the checkpoints if it exists.
- > load_best_model_at_end: Whether or not to load the best model found during training at the end of training.
- > metric_for_best_model: Used in conjunction with load_best_model_at_end to specify the metric to use to compare two different models.
- > early_stopping_patience: Number of epochs with no accuracy improvement over the early_stopping_threshold to wait for stopping.
- > early_stopping_threshold: If accuracy has not become better than this threshold over early_stopping_patience number of epochs, the training will stop.
- > train_val_split_iterator: The dataset can be split many times and the model re-trained in order to obtain more robust estimations of the result. This parameter allows picking the split. It can be any of "train_test_split" where the split is done once, "cross_validate_accuracy" where the split is done multiple times to cross-validate the accuracy. 
- > report_to: The list of integrations to report the results and logs. We use null here as we support no integrations.


### tokenizer_config

- > add_special_tokens: Whether to add special tokens. This is not used anymore.
- > max_length: The maximum length of each tweet.
- > padding: Padding strategy. It can be one of "max_length", "longest", "do_not_pad".
- > truncation: Whether to truncate texts that have length over max_length.
- > return_token_type_ids: Whether to return token type IDs. If left to the default, will return the token type IDs according to the specific tokenizer’s defaul.
- > return_attention_mask: Whether to return the attention mask. If left to the default, will return the attention mask according to the specific tokenizer’s default.

### model_config

The arguments for configuring the basic model properties for the problem of the kaggle competition are provided here:

- >num_labels: number of labels (always 2).
- >problem_type: always use this: "single_label_classification". No other problem types.
- >output_attentions: Whether to output the attention masks along with the output.
- >output_hidden_states: Whether to output the output of the hidden states along with the prediction output.
- >id2label: map of labels, we use this map because some devices are better to use 0,1 rather than 1, -1.{"1": 1,"0": -1}.

### Hyperparameter optimization
While training the parameter under args are regarded as hyperparameters. These parameters can be optimized using the hyperopt library and running multiple trainings. The hyperparameters belong in a set provided with the hyperopt library, so that the maximum accuracy is reached. To activate and use the hyperopt functionality the keys below should be specified inside the json configuration:

- >use_hyperopt: Whether to use the hyperoptimization or not,
- >hyperopt_max_evals: maximum evaluations to run the hyperopt optimization

For each parameter inside args of the json configuration the hyperopt optimization can be specified as:

```json
argsParameter: {"use_hyperopt": <use-hyperoptimization-or-not>, "hyperopt_function":<hyperopt function name>, "arguments": {<arguments of the hyperopt function>}}
```

Example:
```json
argsParameter: {"use_hyperopt": true, "hyperopt_function":"choice", "arguments": {"options": [8,16,32,64]}}
```

- >use_hyperopt: specify as true to optimize this variable inside the interval specified in arguments.
- >hyperopt_function: string name of the hyperopt function.
- >arguments: The arguments of the respective hyperopt function.

Available hyperopt functions are: normal, lognormal, loguniform, qlognormal, qnormal, randint, uniform, uniformint, choice, pchoice.

A full example for roberta can be seen [here](./src/configs/robertaHyperopt.json).

## Reproducing The Report's Work
This assumes that both local and leonhard setup has been done.
```bash
# The models of BERT, RoBERTa, BERTweet and XLNet are trained
# and their submissions are extracted with or without cleaned dataset

cd <path-to-Computational-Intelligence-Lab-directory>
cd src/experimentConfigs
leonhardUsername=<your-leonhard-username>
```

### Create Table I results
```bash
# run the simple training experiments and obtain the baselines
for MODEL_TYPE in $(cat modelsUsed.txt)
do
bash runExperimentOnLeonhard.sh $leonhardUsername /cluster/home/$leonhardUsername/Computational-Intelligence-Lab/src/configs/table1/baselines/$MODE_TYPE.json
sleep 24h # sleep 24 hours until the training is done
MODEL_PATH=../../trainings/$MODEL_TYPE
scp -r $leonhardUsername@login.leonhard.ethz.ch:/Computational-Intelligence-Lab/trainings/$MODEL_TYPE $MODEL_PATH
allModelTrainings=`ls -lrd $MODEL_PATH/*/`
latest_training="${allModelTrainings##* }"
python submission.py load_path=$latest_training batch_size=128 \
text_path=../../data/test_data.txt & # file is in ../../trainings/$MODEL_TYPE/<last-date>/submission.csv
done
```

```bash
# to remove trainings already done if needed
rm -rf ../../trainings/*
```


```bash
# run the simple training experiments of the baselines with preprocessing
for MODEL_TYPE in $(cat modelsUsed.txt)
do
bash runExperimentOnLeonhard.sh $leonhardUsername /cluster/home/$leonhardUsername/Computational-Intelligence-Lab/src/configs/table1/general_preprocessing/$MODEL_TYPE.json
sleep 24h # sleep 24 hours until the training is done
MODEL_PATH=../../trainings/$MODEL_TYPE
scp -r $leonhardUsername@login.leonhard.ethz.ch:/Computational-Intelligence-Lab/trainings/$MODEL_TYPE $MODEL_PATH
allModelTrainings=`ls -lrd $MODEL_PATH/*/`
latest_training="${allModelTrainings##* }"
python submission.py load_path=$latest_training batch_size=128 \
text_path=../../data/test_data.txt & # file is in ../../trainings/$MODEL_TYPE/<last-date>/submission.csv
done
```

### Create Table II results

For general preprocessing see `Create Table I`.
For the other lines:

```bash
for lineName in $(cat ../configs/table2/lineNames.txt)
do
    for MODEL_TYPE in $(cat modelsUsed.txt)
    do
    bash runExperimentOnLeonhard.sh $leonhardUsername /cluster/home/$leonhardUsername/Computational-Intelligence-Lab/src/configs/table2/$lineName/$MODEL_TYPE.json
    sleep 24h # sleep 24 hours until the training is done
    MODEL_PATH=../../trainings/$MODEL_TYPE
    scp -r $leonhardUsername@login.leonhard.ethz.ch:/Computational-Intelligence-Lab/trainings/$MODEL_TYPE $MODEL_PATH
    allModelTrainings=`ls -lrd $MODEL_PATH/*/`
    latest_training="${allModelTrainings##* }"
    python submission.py load_path=$latest_training batch_size=128 \
    text_path=../../data/test_data.txt & # file is in ../../trainings/$MODEL_TYPE/<last-date>/submission.csv
    done
done
```

### Create Table III results

```bash
# Create the reduction up to length 2
for MODEL_TYPE in $(cat modelsUsed.txt)
do
bash runExperimentOnLeonhard.sh $leonhardUsername /cluster/home/$leonhardUsername/Computational-Intelligence-Lab/src/configs/table3/reducelen2/$MODEL_TYPE.json
sleep 24h # sleep 24 hours until the training is done
MODEL_PATH=../../trainings/$MODEL_TYPE
scp -r $leonhardUsername@login.leonhard.ethz.ch:/Computational-Intelligence-Lab/trainings/$MODEL_TYPE $MODEL_PATH
allModelTrainings=`ls -lrd $MODEL_PATH/*/`
latest_training="${allModelTrainings##* }"
python submission.py load_path=$latest_training batch_size=128 \
text_path=../../data/test_data.txt & # file is in ../../trainings/$MODEL_TYPE/<last-date>/submission.csv
done
```

```bash
# Create the reduction up to length 3
for MODEL_TYPE in $(cat modelsUsed.txt)
do
bash runExperimentOnLeonhard.sh $leonhardUsername /cluster/home/$leonhardUsername/Computational-Intelligence-Lab/src/configs/table3/reducelen3/$MODEL_TYPE.json
sleep 24h # sleep 24 hours until the training is done
MODEL_PATH=../../trainings/$MODEL_TYPE
scp -r $leonhardUsername@login.leonhard.ethz.ch:/Computational-Intelligence-Lab/trainings/$MODEL_TYPE $MODEL_PATH
allModelTrainings=`ls -lrd $MODEL_PATH/*/`
latest_training="${allModelTrainings##* }"
python submission.py load_path=$latest_training batch_size=128 \
text_path=../../data/test_data.txt & # file is in ../../trainings/$MODEL_TYPE/<last-date>/submission.csv
done
```

### Create Table IV results

For baselines see table 1 general preprocessing, for spell checking see below:


```bash
# Create the NeuSpell BERT line
for MODEL_TYPE in $(cat modelsUsed.txt)
do
bash runExperimentOnLeonhard.sh $leonhardUsername /cluster/home/$leonhardUsername/Computational-Intelligence-Lab/src/configs/table4/neuspell/$MODEL_TYPE.json
sleep 24h # sleep 24 hours until the training is done
MODEL_PATH=../../trainings/$MODEL_TYPE
scp -r $leonhardUsername@login.leonhard.ethz.ch:/Computational-Intelligence-Lab/trainings/$MODEL_TYPE $MODEL_PATH
allModelTrainings=`ls -lrd $MODEL_PATH/*/`
latest_training="${allModelTrainings##* }"
python submission.py load_path=$latest_training batch_size=128 \
text_path=../../data/test_data.txt & # file is in ../../trainings/$MODEL_TYPE/<last-date>/submission.csv
done
```


### Create Table VI results

```bash
# first line is the baseline with preprocessing and it is the same as in table's 1 second line
# run the simple training experiments of the baselines with preprocessing
for MODEL_TYPE in $(cat modelsUsed.txt)
do
bash runExperimentOnLeonhard.sh $leonhardUsername /cluster/home/$leonhardUsername/Computational-Intelligence-Lab/src/configs/table1/preprocessing/$MODEL_TYPE.json
sleep 24h # sleep 24 hours until the training is done
MODEL_PATH=../../trainings/$MODEL_TYPE
scp -r $leonhardUsername@login.leonhard.ethz.ch:/Computational-Intelligence-Lab/trainings/$MODEL_TYPE $MODEL_PATH
allModelTrainings=`ls -lrd $MODEL_PATH/*/`
latest_training="${allModelTrainings##* }"
python submission.py load_path=$latest_training batch_size=128 \
text_path=../../data/test_data.txt & # file is in $latest_training/submission.csv
done


# second line and third line
# run the simple training experiments of the baselines with preprocessing
mkdir -pv ../../trainings2
for MODEL_TYPE in $(cat modelsUsed.txt)
do
ssh $leonhardUsername@login.leonhard.ethz.ch "rm -rf Computational-Intelligence-Lab/trainings/$MODEL_TYPE/*" # remove trainings from leonhard in order to get the earliest training of the new training
bash runExperimentOnLeonhard.sh $leonhardUsername /cluster/home/$leonhardUsername/Computational-Intelligence-Lab/src/configs/table6/all_layers_finetuning/$MODEL_TYPE.json
sleep 24h # sleep 24 hours until the training is done
MODEL_PATH=../../trainings2/$MODEL_TYPE
scp -r $leonhardUsername@login.leonhard.ethz.ch:/Computational-Intelligence-Lab/trainings/$MODEL_TYPE $MODEL_PATH
allModelTrainings=`ls -ld $MODEL_PATH/*/`
earliest_training="${allModelTrainings##* }"
python submission.py load_path=$earliest_training batch_size=128 \
text_path=../../data/test_data.txt & # file is in $earliest_training/submission.csv
done

# get third line
for MODEL_TYPE in $(cat modelsUsed.txt)
do
MODEL_PATH=trainings/$MODEL_TYPE
allModelTrainings=`ls -lrd $MODEL_PATH/*/`
latest_training="${allModelTrainings##* }"
python submission.py load_path=$latest_training batch_size=128 \
text_path=../../data/test_data.txt & # file is in ../../trainings/bertweet/<latest-date>/submission.
done

```

### Create Table VII results

First generate a data file (combine pos and neg data) and a prediction file. Those two files are the dependencies for this step:

```bash
cd <path-to-Computational-Intelligence-Lab-directory>
for MODEL_TYPE in $(cat src/experimentConfigs/modelsUsed.txt)
do
MODEL_PATH=trainings/$MODEL_TYPE
allModelTrainings=`ls -lrd $MODEL_PATH/*/`
latest_training="${allModelTrainings##* }"
python src/explorations/evaluate_trainset.py load_path=$latest_training batch_size=64 full_or_sub=full-or-sub
python src/explorations/hashtagExperiment.py dataset=full-or-sub load_path=latest_training freq=500 prob=0.7
done
```

### Majority voting results

Download the best submissions from leonhard and use the [notebook](./src/models/majority_voting.ipynb).


## Code Formatting

For code formatting please do these steps:
1. install yapf: `pip install yapf`
2. ~~Download the yapf configuration file at and add it to `<your-repo-path>/.style.yapf`~~
3. Download the pre-commit.sh file from this repo and add it to `<your-repo-path>/.git/hooks/pre-commit`
4. Make sure pre-commit is marked as executable

When committing always enable virtual environment so that the OS can find yapf.
For more information about yapf: See here: https://github.com/google/yapf

Guideline: Do any contributions to yapf but always announce your decisions to the team.


## Repo structure

The most important parts of this repository structure are presented below.


├── [data](./data/) Here the data provided by the kaggle competition are stored.

├── [doc](./doc/) Documentation related files are put inside here.

│   └── documentsToRead.csv External documents(papers/articles/etc) that help in developing this solution.

│   └── report.tex a minimum report file for this solution. This will be later used in the submission of the project. 

├── [src](./src) Here is all the source code of the group's solution

├── [configs](./configs/) Here are the configuration files for running trainings of models.

│   ├── [experimentConfigs](./src/experimentConfigs/) Here are the experiment configurations. There are many json files in here each describing a series of experiments. This helps identify which are the best combinations of parameters/models/preprocessors to solve the problem.

│   ├── [explorations](./src/explorations) Here are all the methods and strategies tried before making them models or methods in 

│   ├── [models](./src/models/) Various models used in this repository. Mainly the transformersModel.py is used.

<!-- │   │   ├── [Model.py](./src/models/) An abstract model used in this framework to execute a model. Since models are TF, pytorch or any other frameworks models, a wrapper class is needed to be constructed as interface to call their trainining functions and to test them in the problem. 

│   │   ├── [bagOfWords2LayersModel.py](./src/models/bagOfWords2LayersModel.py) A simple bag of words model with 2 non-linear (relu) layers.

│   │   ├── [modelMaps.py](./src/models/modelMaps.py) A mapping for models and their constructors. Currently this is only for transformer models. 

│   │   ├── [transformersModel.py](./src/models/transformersModel.py) A generic wrapper of transformer models (at least those that work) for retraining these models in the tweeter dataset.

│   │   └── [trivialModel.py](./src/models/trivialModel.py) This is a trivial model to test that `Model.py` and its inheritance work. 

│   ├── [notebooks] (./src/notebooks/) Notebooks that are used inside this project for a more intuitive/visualized perspective of the tools developed.  -->

│   ├── [preprocessing](./src/preprocessing/) The preprocessing of input as tokenization is implemented here. Each file contains a different approach and it is likely to be connected with a model inside the models folder. 

<!-- │   │   ├── [InputPipeline.py](./src/preprocessing/InputPipeline.py) An abstract pipeline of preprocessing the input. Preprocessing can work with TF, pytorch or any other frameworks used in models. This class interfaces the input preprocessing and it can be used by a facade class in order to train any model present in this framework.

│   │   ├── [bagOfWordsPipeline.py](./src/preprocessing/bagOfWordsPipeline.py) A simple bag of words tokenizer. This works with `bagOfWords2LayersModel`.

│ │ ├── [pipelineMaps.py](./src/preprocessing/pipelineMaps.py) A mapping for tokenizers and their constructors.
Currently this is only for transformer tokenizers.

│ │ ├── [pretrainedTransformersPipeline.py](./src/preprocessing/pretrainedTransformersPipeline.py) A generic wrapper for
reading the tweeter dataset for the pytorch or the tensorflow framework. It works with `transformersModel` wrapper. -->

├── [tests](./tests/) test bash scripts or integration tests in any language are put here.

├── [trainings](./trainings/) Here the training of each model are stored to a specific folder. This folder is created by the user and is not present in Github repository when downloaded.

## Dependencies

To run the code on ETHz's HPC, the modules needed are loaded as:

```bash
module load gcc/6.3.0 python_gpu/3.8.5 hdf5/1.10.1 eth_proxy
```

See [setup_leonhard.sh](./setup_leonhard.sh), [setup_environment.sh](./setup_environment.sh), and [requirements.txt](./requirements.txt).

## Cluster training requirements
All trainings need to be submitted to the 24 hour queue:
- 8+ cpu cores
- 10+ GB of RAM
- 1 GPU with 10GB+ VRAM, Typically this would be a GeForceRTX2080Ti


## Developers - Students

- He Liu
- Ioannis Athanasiadis
- Levin Moser

## Acknowledgements

- ETHZ for providing the leonhard cluster nodes to us
- Huggingface for their transformers library models
