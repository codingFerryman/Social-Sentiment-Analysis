# Computational Intelligence Lab Project 2
This is the project 2 of CIL course in ETHZ.

## Developing
To work for this project create a virtual environment in a folder outside the repo-path or use .gitignore to ignore any files of the virtual envronment folder.

### Code Formatting

For code formatting please do these steps:
1. install yapf: `pip install yapf`
2. ~~Download the yapf configuration file at and add it to `<your-repo-path>/.style.yapf`~~
3. Download the pre-commit.sh file from this repo and add it to `<your-repo-path>/.git/hooks/pre-commit`
4. Make sure pre-commit is marked as executable

When commiting always enable virtual environment so that the OS can find yapf.
For more information about yapf: See here: https://github.com/google/yapf

Do any contributions to yapf but always anounce your decisions to the team.


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

## Execute on HPC
The following command will request 1 GPU core, 8GB RAM, and 1 GPU with 10GB+ VRAM for 23 hours:
```bash
bsub -W 23:00 -n 1 -R "rusage[mem=8192,ngpus_excl_p=1]" -R "select[gpu_mtotal0>=10240]" \
source venv/bin/activate \
venv/bin/python src/experimentConfigs/experiment.py test_path=$TRAINING_JSON_CONFIG_PATH report_path=$REPORT_JSON_PATH
```
(Assume that you are running from project's root directory.)

Please make sure that the proxy http://proxy.ethz.ch:3128 has been set for downloading external models.
