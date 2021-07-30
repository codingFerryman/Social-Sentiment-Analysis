#
# This script creates a submission csv file to be submitted to the kaggle competition.
# It loads an already trained model from its checkopoints. The device and the model's batch size
# can be also specified in this file. 
# 
# A typical example of usage for roberta is:
#
# python submission.py load_path=../../trainings/roberta-base/20210709-102233 batch_size=128 \
# text_path=../../data/test_data.txt
#
import os
import sys
from pathlib import Path

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from utils import loggers, set_seed
from utils import get_data_path
from transformersPredict import TransformersPredict
from transformersPredictWithHashtag import TransformersPredictWithHashtag

logger = loggers.getLogger("PredictForSubmission", True)
set_seed()


def main(args: list):
    """
    The main function of submission. It will predict the sentiment of texts in the text data
    Args:
        args (list): a dictionary containing the program arguments (sys.argv)
        - load_path: The root directory containing 'model' and 'tokenizer'
        - batch_size: The batch size in prediction. The default is 256
        - device: The index of cuda device for prediction.
            If not given, the program will automatically use the first cuda device otherwise the cpu
        - hashtag_analysis: Use hashtag information or not in submission. Enable by default.
        - fast_tokenizer: Use Fast Tokenizer or not in predictions. Better to use the same as training tokenizer
            Using normal tokenizer by default
        - text_path: The text file to be processed. data/test_data.txt is used by default.

    """
    argv = {a.split('=')[0]: a.split('=')[1] for a in args[1:]}

    load_path = argv.get('load_path', None)
    assert load_path, "No load_path specified"

    batch_size = argv.get('batch_size', 128)

    device = argv.get('device', None)

    text_path = argv.get('text_path', None)

    hashtag_analysis = argv.get('hashtag', 'true').lower()
    assert hashtag_analysis in ['true', 'false']
    hashtag_analysis = False if 'f' in hashtag_analysis else True

    fast_tokenizer = argv.get('fast_tokenizer', 'false').lower()
    assert fast_tokenizer in ['true', 'false']
    fast_tokenizer = False if 'f' in fast_tokenizer else True

    if text_path is None:
        data_path = get_data_path()
        _text_path = Path(data_path, 'test_data.txt')
        if _text_path.is_file():
            text_path = _text_path
        else:
            logger.error("No text_path specified")
            exit(0)

    logger.info(f"Predicting sentiment from data inside {text_path}")

    if not hashtag_analysis:
        trans_predict = TransformersPredict(load_path=load_path, text_path=text_path, device=device,
                                            fast_tokenizer=fast_tokenizer)
    else:
        trans_predict = TransformersPredictWithHashtag(load_path=load_path, text_path=text_path, device=device,
                                                       fast_tokenizer=fast_tokenizer)
    trans_predict.predict(batch_size=batch_size)
    trans_predict.submissionToFile()


if __name__ == '__main__':
    main(sys.argv)
    # load_path = "/home/he/Workspace/cil-project/trainings/vinai/bertweet-base/20210721-024602"
    # text_path = "/home/he/Workspace/cil-project/data/test_data.txt"
    # trans_predict = TransformersPredict(load_path, text_path)
    # trans_predict.predict()
    # trans_predict.submissionToFile()
