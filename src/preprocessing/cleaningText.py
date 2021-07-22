import os
import sys
from pathlib import Path
from typing import Dict, Callable, Union

import modin.pandas as mpd
import neuspell
import pandas as pd
import regex
import torch
from cleantext import clean
from distributed import Client
from neuspell import BertChecker
from nltk.tokenize.treebank import TreebankWordDetokenizer
from tqdm import tqdm

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from utils.loggers import getLogger
from utils.others import get_project_path, get_data_path

logger = getLogger("CleaningText", True)
PROJECT_PATH = get_project_path()
DATA_PATH = get_data_path()
# ProgressBar.enable()

EMOTICONS = r"""
    (?:
      [<>]?
      [:;=8]                     # eyes
      [\-o\*\']?                 # optional nose
      [\)\]\(\[dDpP/\:\}\{@\|\\] # mouth
      |
      [\)\]\(\[dDpP/\:\}\{@\|\\] # mouth
      [\-o\*\']?                 # optional nose
      [:;=8]                     # eyes
      [<>]?
      |
      <3                         # heart
    )"""

REGEXPS = (
    # ASCII Emoticons
    EMOTICONS,
    # ASCII Arrows
    r"""[\-]+>|<[\-]+""",
    # Twitter hashtags:
    r"""(?:\#+[\w_]+[\w\'_\-]*[\w_]+)""",
    # Remaining word types:
    r"""
    (?:[^\W\d_](?:[^\W\d_]|['\-_])+[^\W\d_]) # Words with apostrophes or dashes.
    |
    (?:[+\-]?\d+[,/.:-]\d+[+\-]?)  # Numbers, including fractions, decimals.
    |
    (?:[\w_]+)                     # Words without apostrophes or dashes.
    |
    (?:\.(?:\s*\.){1,})            # Ellipsis dots.
    |
    (?:\S)                         # Everything else that isn't whitespace.
    """,
)

WORD_RE = regex.compile(r"""(%s)""" % "|".join(REGEXPS), regex.VERBOSE | regex.I | regex.UNICODE)

# WORD_RE performs poorly on these patterns:
HANG_RE = regex.compile(r"([^a-zA-Z0-9])\1{3,}")

# The emoticon string gets its own regex so that we can preserve case for
# them as needed:
EMOTICON_RE = regex.compile(EMOTICONS, regex.VERBOSE | regex.I | regex.UNICODE)


def reduce_lengthening(text, reduce_to_length: int = 3):
    pattern = regex.compile(r"(.)\1{2,}")
    return pattern.sub(r"\1" * reduce_to_length, text)


def cleaning_default(text: Union[str, list]):
    to_be_removed = r'(<.*?>)|[\'\"]'
    if type(text) is str:
        return regex.sub(to_be_removed, '', text.strip())
    else:
        _tmp = pd.Series(text)
        _tmp = _tmp.str.strip()
        _result = _tmp.str.replace(to_be_removed, '').to_list()
        return _result


def cleaning_masks(text: Union[str, list]):
    to_be_removed = r'(<.*?>)|(\.{3})|(http[^a-zA-Z])'
    if type(text) is str:
        return regex.sub(to_be_removed, '', text.strip())
    else:
        _tmp = pd.Series(text)
        _tmp = _tmp.str.strip()
        _result = _tmp.str.replace(to_be_removed, '').to_list()
        return _result


def cleaning_strip(text: Union[str, list]):
    if type(text) is str:
        return text.strip()
    else:
        _tmp = pd.Series(text)
        _result = _tmp.str.strip().to_list()
        return _result

def _cleaning_tweet(text: str, spell_checker=None):
    dtknzr = TreebankWordDetokenizer()
    if spell_checker:
        text = spell_checker(text)
    text = dtknzr.detokenize(text.split())
    text = cleaning_masks(text)
    text = clean(text,
                 fix_unicode=True,  # fix various unicode errors
                 to_ascii=True,  # transliterate to closest ASCII representation
                 no_line_breaks=True,  # fully strip line breaks as opposed to only normalizing them
                 no_urls=True,  # replace all URLs with a special token
                 no_phone_numbers=True,  # replace all phone numbers with a special token
                 no_currency_symbols=True,  # replace all currency symbols with a special token
                 replace_with_url="",
                 replace_with_phone_number="",
                 replace_with_currency_symbol="",
                 lang="en"
                 )
    text = reduce_lengthening(text, 2)
    # Shorten problematic sequences of characters
    safe_text = HANG_RE.sub(r"\1\1\1", text)
    # Tokenize:
    words = WORD_RE.findall(safe_text)
    text = " ".join(words)
    return text


def cleaning_tweet(text_list, check_spell=True, batch_size=512, is_test=False):
    if type(text_list) is str:
        is_test = True
        text_list = [text_list]
    if check_spell is True:
        spell_checker_path = Path(PROJECT_PATH, 'src', 'preprocessing', 'subwordbert-probwordnoise')
        spell_checker_exists = spell_checker_path.exists()
        if Path().resolve().parts[1] == 'cluster' and not spell_checker_exists:
            logger.info("Set the proxy for downloading spell checker")
            proxy = "http://proxy.ethz.ch:3128"
            os.environ['http_proxy'] = proxy
            os.environ['HTTP_PROXY'] = proxy
            os.environ['https_proxy'] = proxy
            os.environ['HTTPS_PROXY'] = proxy
        if not spell_checker_exists:
            logger.info("Downloading the spell checker ...")
            neuspell.seq_modeling.downloads.download_pretrained_model(str(spell_checker_path))
        else:
            logger.info("The pre-trained spell checker already exists.")

        if torch.cuda.is_available():
            checker = BertChecker(device='cuda')
        else:
            checker = BertChecker(device='cpu')
        checker.from_pretrained(spell_checker_path)
        logger.info("Correcting misspelling words ...")
        results = []
        if is_test:
            for test_sent in text_list:
                _id = test_sent.split(',', 1)[0]
                _sent = test_sent.split(',', 1)[-1]
                _sent_cleaned = checker.correct(_sent)
                results.append(','.join([_id, _sent_cleaned]))
        else:
            for i in tqdm(range(0, len(text_list), batch_size)):
                text_batch = text_list[i:i + batch_size]
                text_batch = checker.correct_strings(text_batch)
                results.extend(text_batch)
                torch.cuda.empty_cache()
        text_list = results
    if is_test:
        text = []
        for test_sent in text_list:
            _id = test_sent.split(',', 1)[0]
            _sent = test_sent.split(',', 1)[-1]
            _sent_cleaned = _cleaning_tweet(_sent)
            _result = ",".join([str(_id), _sent_cleaned])
            text.append(_result)
    else:
        logger.info("Cleaning text by 3 workers. It will take around 20 min, please wait ...")
        client = Client(n_workers=3)
        _tmp = mpd.Series(text_list)
        _tmp = _tmp.map(_cleaning_tweet)
        text = _tmp.to_list()
    return text


def cleaningMap() -> Dict[str, Callable]:
    return {
        "default": cleaning_default,
        "masks": cleaning_masks,
        "strip": cleaning_strip,
        "tweet": cleaning_tweet
    }


def main(args: list):
    """The function for cleaning data and export the cleaned data to a new file"""
    argv = {a.split('=')[0]: a.split('=')[1] for a in args[1:]}
    data_path = argv.get('data_path', Path(DATA_PATH, 'test_data.txt'))
    assert data_path is not None, "No data_path specified"
    input_path = Path(data_path)
    input_file = input_path.parts[-1]
    input_file_name = input_file.split('.')[0]
    input_file_extension = input_file.split('.')[-1]
    output_file = input_file_name + '_clean.' + input_file_extension
    output_dir = input_path.parents[0]
    output_path_default = Path(output_dir, output_file).resolve()
    output_path = argv.get('output', str(output_path_default))

    with open(input_path, 'r') as fr:
        input_data = fr.readlines()
    input_data = cleaning_strip(input_data)
    # input_data = list(set(input_data))
    input_data = list(filter(None, input_data))

    cleaned = cleaning_tweet(input_data, is_test=True if 'test' in input_file_name else False)
    cleaned_lines = [t + '\n' for t in cleaned]

    with open(Path(output_path), 'w') as fw:
        fw.writelines(cleaned_lines)


if __name__ == '__main__':
    main(sys.argv)
