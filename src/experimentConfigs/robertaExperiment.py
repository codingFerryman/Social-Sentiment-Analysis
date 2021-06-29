import os
import sys
import time
from pathlib import PurePath

from sklearn.model_selection import train_test_split

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from preprocessing import PretrainedTransformersPipeLine
from models import TransformersModel

os.environ['CUDA_VISIBLE_DEVICES'] = "1"
# Load the model
model_name = 'cardiffnlp/twitter-roberta-base-sentiment'
pipeline = PretrainedTransformersPipeLine(model_name)
pipeline.loadData(ratio='sub')
encDataTrain, encDataVal = pipeline.getEncodedDataset(splitter=train_test_split, test_size=0.1)
model = TransformersModel(modelName_or_pipeLine=pipeline)
eval_log = model.trainModel(freeze_model=False)
trainer = model.getTrainer()
# %%

trainer.save_model(PurePath('/trainings', 'model', model_name, time.strftime("%Y%m%d-%H%M%S")))

# %%
#
# # Make predictions
# test_loader = DataLoader(test_text, batch_size=64)
# predictions = torch.tensor([], device=device)
# with torch.no_grad():
#     for test_data in test_loader:
#         inputs = tokenizer.batch_encode_plus(truncation=True, padding=True, max_length=max_test)
#         inputs = inputs.to(device)
#         logit = model(**inputs).logits
#         prediction = torch.argmax(torch.softmax(logit,dim=-1), dim=-1)
#         predictions = torch.cat((predictions, prediction), 0)
#
# #%%
#
# # Make the predictions be compatible with the submission
# pred = predictions.int().tolist()
# # pred = np.where(pred==0, -1, pred)
# pred_id = test_id+zero_len_idx_test
# pred_est = pred+[random.choice([0,1]) for i in range(len(zero_len_idx_test))]
# pred_est = [p if p==1 else -1 for p in pred_est]
# pred_dict = {'Id': pred_id, 'Prediction': pred_est}
# pred_df = pd.DataFrame(pred_dict)
# pred_df.to_csv('./submission.csv', index=False)
