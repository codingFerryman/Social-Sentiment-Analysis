import os
import sys
from pathlib import Path

from tqdm.auto import tqdm

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from utils.others import get_data_path

DATA_PATH = get_data_path()

if Path().resolve().parts[1] == 'cluster':
    import stanza

    proxy = "http://proxy.ethz.ch:3128"
    os.environ['HTTP_PROXY'] = proxy
    os.environ['HTTPS_PROXY'] = proxy

    os.environ['STANZA_RESOURCES_DIR'] = os.path.join(os.getenv("SCRATCH"), '.cache/stanza_resources/')
    stanza.download(lang='en', model_dir=os.getenv('STANZA_RESOURCES_DIR'))
    nlp = stanza.Pipeline(lang='en', dir=os.getenv('STANZA_RESOURCES_DIR'), processors='tokenize,pos,lemma')

else:
    import stanza

    stanza.download('en')
    nlp = stanza.Pipeline(lang='en', processors='tokenize,pos,lemma')


def lemmatization(text):
    doc = nlp(text)
    _result = " ".join([word.lemma for sent in doc.sentences for word in sent.words])
    return _result


def combine_data(*data_path):
    _tmp = []
    with open(data_path[0], 'r') as fp:
        _data = fp.readlines()
        _data_processed = []
        for _t in tqdm(_data, dynamic_ncols=True, desc="Lemmatization", mininterval=30, maxinterval=60):
            _data_processed.append(lemmatization(_t) + '\n')
        _tmp.append(_data_processed)
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
    output_path = Path(DATA_PATH, 'train_neg_full_aug.txt')
    with open(output_path, 'w') as fp:
        fp.writelines(result)

    combine_path_list = [
        Path(DATA_PATH, 'train_pos_full.txt'),
        Path(DATA_PATH, 'train_pos_full_clean.txt')
    ]
    result = combine_data(*combine_path_list)
    output_path = Path(DATA_PATH, 'train_pos_full_aug.txt')
    with open(output_path, 'w') as fp:
        fp.writelines(result)
