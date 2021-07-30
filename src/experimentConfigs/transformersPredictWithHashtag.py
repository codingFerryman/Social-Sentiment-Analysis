import os
import random
import sys
from pathlib import Path

import pandas as pd

from explorations.hashtagExperiment import load_hashtag_config, hashtag_matters
from transformersPredict import TransformersPredict

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from utils import loggers, set_seed, get_project_path

set_seed()
logger = loggers.getLogger("transformersPredictWithHashtag", True)
PROJECT_DIRECTORY = get_project_path()


class TransformersPredictWithHashtag(TransformersPredict):
    def __init__(self, freq_threshold: int = 500, prob_threshold: float = 0.7, **kwargs):
        """ This is the transformers prediction class with hashtag analysis.
        It manages the prediction of data given a model's checkpoint (load_path) and data to predict (text_path).
        The device to use can be also specified, by default 'cuda:0' is used or 'cpu' (if there is no gpu present).

        Args:
            freq_threshold: The frequency threshold. Hashtags with lower frequency will be ignored.
            prob_threshold: The probability/ratio threshold. Hashtags with lower probability/ratio will be ignored.
            load_path (Path): The root directory containing 'model' and 'tokenizer'. This is common with the torch checkpoint structure.
            text_path (Path): The text file to be processed. data/test_data.txt by default.
            fast_tokenizer (bool, optional): Whether to use a fast tokenizer. Some models may output an error when this flag is set to false. Defaults to False.
            device (str, optional): String identifier of the device to be used. Defaults to None and 'cuda:0' is used or if there is no gpu present 'cpu'
            is_test (bool, optional): Whether the data are comming from the test data provided by the kaggle competition or not. Defaults to True.
        """
        super(TransformersPredictWithHashtag, self).__init__(**kwargs)

        self.hashtag_dict = load_hashtag_config()
        self.freq_threshold = freq_threshold
        self.prob_threshold = prob_threshold

    def predict(self, batch_size=128):
        super(TransformersPredictWithHashtag, self).predict(batch_size)
        id_zero_len = self.data['zero_len_ids']

        pred_id = self.data['ids'] + id_zero_len
        pred_scores = self.get_scores().tolist() + [0.5 for _ in range(len(id_zero_len))]
        pred_labels = self.get_predictions().tolist() + [random.choice([-1, 1]) for _ in range(len(id_zero_len))]
        pred_text = [self.data['text'][int(idx) - 1] for idx in pred_id]

        pred_df = pd.DataFrame({'id': pred_id, 'score': pred_scores, 'prediction': pred_labels, 'text': pred_text})
        pred_df = pred_df.astype({
            'id': int,
            'score': float,
            'prediction': int,
            'text': str
        })
        self.pred_df = hashtag_matters(pred_df, freq_threshold=self.freq_threshold, prob_threshold=self.prob_threshold)

    def submissionToFile(self, save_path: Path = None):
        """ This combine prediction probabilities with HashTag polarities,
        then puts the predictions already predicted to a csv file ready
        for the kaggle api to read and produce the final results from.

        Args:
            save_path (Path, optional): csv file path to put the predictions inside. Defaults to self.laod_path / submission.csv .
        """
        if save_path is None:
            save_path = Path(self.load_path, 'submission.csv')

        pred_dict = {'Id': self.pred_df.id, 'Prediction': self.pred_df.new_prediction}
        pred_df = pd.DataFrame(pred_dict)
        pred_df = pred_df.astype({
            'Id': int,
            'Prediction': int
        })
        pred_df.sort_values('Id', inplace=True)
        pred_df.to_csv(save_path, index=False)

        logger.info('The submission file has been saved to ' + str(save_path))
