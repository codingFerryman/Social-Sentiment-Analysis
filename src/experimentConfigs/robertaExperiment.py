import json
import os
from pathlib import Path

from models import TransformersModel
# sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from submission import TransformersPredict
from utils import get_project_path
import torch

project_directory = get_project_path()

if torch.cuda.is_available():
    cuda_rank = 0
else:
    cuda_rank = -1
os.environ['CUDA_VISIBLE_DEVICES'] = str(cuda_rank)

# Load the config
with open(Path(project_directory, 'src', 'configs', 'roberta_base_debug.json'), 'r') as fp:
    config = json.load(fp)

# Load the model
model_name_or_path = config['model_name_or_path']
model = TransformersModel(modelName_or_pipeLine=model_name_or_path,
                          fast_tokenizer=config.get('fast_tokenizer'))

# Register metrics
if type(config['metric']) is str:
    config['metric'] = [config['metric']]
model.registerMetric(*config['metric'])

# Load training data
model.loadData(ratio=d['data_load_ratio'])

# Train the model and get evaluation history
eval_log = model.trainModel(
    train_val_split_iterator=config['args'].get('train_val_split_iterator', "train_test_split"),
    model_config=config['model_config'],
    tokenizer_config=config['tokenizer_config'],
    trainer_config=config['args']
)

# Save the model
model.save()

# Get the best result
best_model_metric = model.getBestMetric()

# Get the path of the model
model_saved_path = model.training_saving_path

# Load back the trainer
trainer = model.getTrainer()

# Predict the test data
model_predict = TransformersPredict(load_path=model_saved_path,
                                    text_path=Path(project_directory, 'data', 'test_data.txt'),
                                    cuda_device=cuda_rank)
model_predict.predict(batch_size=32)

model_predict.submission_file(Path(model_saved_path, 'submission.csv'))
