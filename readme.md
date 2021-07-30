# Computational Intelligence Lab Project 2
This is our project ETHZ CIL Text Classification 2021 of CIL course in ETH Zurich.



## Results

- Results on validation data can be seen [here](https://supernlogn.github.io/Computational-Intelligence-Lab/)
- Results on test data can be seen in the respective Kaggle competition.

## Setup
To work with this project a certain procedure is needed. First clone this repo. Then we recommend using the automated scripts for setup. Thsese scripts create a virtual environment and download any data needed for this repo. Please be inside the ETHZ network while using them, to download the dataset.

Use an automated script:
```bash
cd <path-to-Computational-Intelligence-Lab-directory>
# gpu local
bash setup_local.sh
# leonhard cluster
bash setup_leonhard.sh
# Windows machine
".\setup_local_windows.ps1"
```




or do it manually:
```bash
cd <path-to-Computational-Intelligence-Lab-directory>
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
There are many scripts and ways of execution that were tried during the development of this project. Here we list all of them, in order to execute them you need to be in their directory. Below we also include examples of runnning them. [ [Skip to examples](https://github.com/supernlogn/Computational-Intelligence-Lab#experimentation-on-leonhard) ]


- >bash [runExperimentOnLeonhard.sh](./src/experimentConfigs/runExperimentOnLeonhard.sh) `<leonhard-username> <path-to-configuration-inside-Leonhard>` This script submits a training of a model to run by the leonhard cluster in under 24 hours using a gpu. The model and the preprocessing are described by the configuration file provided. After running the training the validation results are stored inside the [report.json](./docs/report.json) and uploaded, so that they can be viewed in web.
- >bash [runExperimentAndUploadReportCluster.sh](./src/experimentConfigs/runExperimentAndUploadReportCluster.sh) `<path-to-configuration-inside-Leonhard>`  This script is executed inside the cluster. Its purpose is to load the modules and environment needed by the scripts to run, execute the training and then upload the report of the training. This should be executed inside the cluster using the bsub command. To submit this while inside the cluster you can type:
- >bash [runExperimentAndUploadReport.sh](./src/experimentConfigs/runExperimentAndUploadReport.sh) `<path-to-configuration-inside-your-pc>` This script loads the environment needed to execute the experiment.py and then executes the training of the model. After the training ends, the results are stored inside docs/report.json and uploaded to github This scripts also assumes that a cil-venv environment exists in the bas of the repo and tries to load it.
- >bash [uploadNewReport.sh](./src/experimentConfigs/uploadNewReport.sh) `<report-file-to-upload =../../docs/report.json>`  This scripts uploads the report.json to github by commiting it and pushing to the branch that the local github project is currently on. It accepts one argument the path of the report file. The argument by default is `../../docs/report.json`.
- >`python` [experiment.py](./src/experimentConfigs/experiment.py) `test_path=<path-to-configuration> report_path=<path-to-the-report.json-file>` This script provides an entry to the framework and launches an experiment(training) from a configuration file. The configuration file is a json file containing configuration for the model and the preprocessing. The validation results of the training are stored inside the json file specified. If this is not specified `docs/report.json` is used by default. Args for this script are the `test_path` for setting the path of the configuration json file and the `report_path` for setting the path for the report json file to be written or appended.
- >`python` [submission.py](./src/experimentConfigs/submission.py) `load_path=<directory-containing-'model'-and-'tokenizer'>  batch_size=<batch-size-used-for-the-model=128>  device=<gpu-id|cpu-id =cuda:0|cpu> text_path=<path-where-test_data.txt is =../../data/test_data.txt>` This script creates a submission csv file to be submitted to the kaggle competition. It loads an already trained model from its checkpoints. The device and the model's batch size can be also specified when calling it.
- >`python` [clusterFeatures.py](./src/explorations/clusterFeatures.py) This script is used to view the feature (output of last layers) vectors of a model after performing PCA on them. 
- >`python` [cleaningText.py](./src/preprocessing/cleaningText.py) `data_path=<path-text-file-with-a-tweet-per-line> output=<path-to-output-cleaned-text-file>` This script cleans/preprocess the text from a file and exports the cleaned data to another (new) file.
- >`python` [hastagExperiment.py](./src/experimentConfigs/hastagExperiment.py) `dataset=<use-full-or-partial-dataset> load_path=<trained-model-path> freq=500 prob=0.7` This script extracts hashtags and finds how much a hashtag matters. It is used to produce the hashtag analysis in the project's final report.

### Experimentation on Leonhard
Run these only after completing the setup section.

From your pc's console:
```bash
cd <path-to-Computational-Intelligence-Lab-directory>
cd src/experimentConfigs
bash runExperimentOnLeonhard.sh <leonhard-username> <path-to-configuration-inside-Leonhard> report_path=../../docs/report.json
```

Typical paths to configuration:
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

## Configurations
The configuration for each training are stored inside a json file. See [example](./src/configs/roberta_base.json).


## Reproducing The Report's Work



### Code Formatting

For code formatting please do these steps:
1. install yapf: `pip install yapf`
2. ~~Download the yapf configuration file at and add it to `<your-repo-path>/.style.yapf`~~
3. Download the pre-commit.sh file from this repo and add it to `<your-repo-path>/.git/hooks/pre-commit`
4. Make sure pre-commit is marked as executable

When commiting always enable virtual environment so that the OS can find yapf.
For more information about yapf: See here: https://github.com/google/yapf

Guideline: Do any contributions to yapf but always anounce your decisions to the team.


## Repo structure

The most important parts of this repository structure are presented below.


├── [data](./data/) Here the data provided by the kaggle competition are stored.

├── [doc](./doc/) Documentation related files are put inside here.

│   └── documentsToRead.csv External documents(papers/articles/etc) that help in developing this solution.

│   └── report.tex a minimum report file for this solution. This will be later used in the submission of the project. 

├── [src](./src) Here is all the source code of the group's solution

│   ├── [experimentConfigs](./src/experimentConfigs/) Here are the experiment configurations. There are many json files in here each describing a series of experiments. This helps identify which are the best combinations of parameters/models/preprocessors to solve the problem. 

│   ├── [models](./src/models/) Various models used in this repository.

│   │   ├── [Model.py](./src/models/) An abstract model used in this framework to execute a model. Since models are TF, pytorch or any other frameworks models, a wrapper class is needed to be constructed as interface to call their trainining functions and to test them in the problem. 

│   │   ├── [bagOfWords2LayersModel.py](./src/models/bagOfWords2LayersModel.py) A simple bag of words model with 2 non-linear (relu) layers.

│   │   ├── [modelMaps.py](./src/models/modelMaps.py) A mapping for models and their constructors. Currently this is only for transformer models. 

│   │   ├── [transformersModel.py](./src/models/transformersModel.py) A generic wrapper of transformer models (at least those that work) for retraining these models in the tweeter dataset.

│   │   └── [trivialModel.py](./src/models/trivialModel.py) This is a trivial model to test that `Model.py` and its inheritance work. 

│   ├── [notebooks] (./src/notebooks/) Notebooks that are used inside this project for a more intuitive/visualized perspective of the tools developed. 

│   ├── [preprocessing](./src/preprocessing/) The preprocessing of input as tokenization is implemented here. Each file contains a different approach and it is likely to be connected with a model inside the models folder. 

│   │   ├── [InputPipeline.py](./src/preprocessing/InputPipeline.py) An abstract pipeline of preprocessing the input. Preprocessing can work with TF, pytorch or any other frameworks used in models. This class interfaces the input preprocessing and it can be used by a facade class in order to train any model present in this framework.

│   │   ├── [bagOfWordsPipeline.py](./src/preprocessing/bagOfWordsPipeline.py) A simple bag of words tokenizer. This works with `bagOfWords2LayersModel`.

│ │ ├── [pipelineMaps.py](./src/preprocessing/pipelineMaps.py) A mapping for tokenizers and their constructors.
Currently this is only for transformer tokenizers.

│ │ ├── [pretrainedTransformersPipeline.py](./src/preprocessing/pretrainedTransformersPipeline.py) A generic wrapper for
reading the tweeter dataset for the pytorch or the tensorflow framework. It works with `transformersModel` wrapper.

├── [tests](./tests/) test bash scripts or integration tests in any language are put here.

├── [trainings](./trainings/) Here the training of each model are stored to a specific folder.

## Dependencies

To run the code on ETHz's HPC, please load the module as:

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