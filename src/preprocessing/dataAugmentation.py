import os
import sys
from pathlib import Path

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from utils.others import get_data_path

DATA_PATH = get_data_path()


def combine_data(*data_path):
    _tmp = []
    for file_path in data_path:
        with open(file_path, 'r') as fp:
            _tmp.append(fp.readlines())
    _result = list(set().union(*_tmp))
    return _result


if __name__ == '__main__':
    combine_path_list = [
        Path(DATA_PATH, 'train_neg_full.txt'),
        Path(DATA_PATH, 'train_neg_full_clean.txt')
    ]
    result = combine_data(*combine_path_list)
    output_path = Path(DATA_PATH, 'train_neg_aug.txt')
    with open(output_path, 'w') as fp:
        fp.writelines(result)

    combine_path_list = [
        Path(DATA_PATH, 'train_pos_full.txt'),
        Path(DATA_PATH, 'train_pos_full_clean.txt')
    ]
    result = combine_data(*combine_path_list)
    output_path = Path(DATA_PATH, 'train_pos_aug.txt')
    with open(output_path, 'w') as fp:
        fp.writelines(result)
