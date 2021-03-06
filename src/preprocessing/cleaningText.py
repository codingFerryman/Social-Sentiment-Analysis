import multiprocessing
import os
import string
import sys
from pathlib import Path
from typing import Callable, Union, List

import pandas as pd
import regex
import torch
from cleantext.clean import clean
from joblib import Parallel, delayed
from tqdm import tqdm

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from utils.loggers import getLogger
from utils.others import get_project_path, get_data_path

logger = getLogger("CleaningText", True)
PROJECT_PATH = get_project_path()
DATA_PATH = get_data_path()


def reduce_lengthening(text: str, reduce_to_length: int = 3):
    """ reduces the lengths of sequences of the same character like 'ooooo' to the provided reduce_to_length

    Args:
        text (str): sentence text
        reduce_to_length (int, optional): maximum length to reduce sequences of repeated characters. Defaults to 3.

    Returns:
        str: sentece text containing sequences of repeated characters of length up to reduce_to_length
    """
    pattern = regex.compile(r"(.)\1{2,}")
    return pattern.sub(r"\1" * reduce_to_length, text)


def cleaning_default(text: Union[str, list], **kwargs):
    """Default cleaning removes http:/ or characters from string
    <user>, <url>, ... are already removed during data loading

    Args:
        text (Union[str, list]): [description]

    Returns:
        Union[str, List[str]]: the same type as text
    """
    to_be_removed = r'(http[^a-zA-Z])'
    if type(text) is str:
        return regex.sub(to_be_removed, '', text.strip())
    else:
        _tmp = pd.Series(text)
        _tmp = _tmp.str.strip()
        _result = _tmp.str.replace(to_be_removed, '').to_list()
        return _result


def cleaning_strip(text: Union[str, List[str]], **kwargs) -> Union[str, List[str]]:
    """ strips text if text is str or strip every text element in list if it is a list

    Args:
        text (Union[str, List[str]): list of texts to strip

    Returns:
        Union[str, List[str]]: the same type as text (the text is stripped)
    """
    if type(text) is str:
        return text.strip()
    else:
        _tmp = pd.Series(text)
        _result = _tmp.str.strip().to_list()
        return _result


def _remove_punct(text: str, keep_neutral=False):
    to_be_removed = string.punctuation
    neutral_punct = set('!\'#-?')
    if keep_neutral:
        _tmp = set(string.punctuation) - neutral_punct
        to_be_removed = "".join(_tmp)
    return text.translate(str.maketrans('', '', to_be_removed))


def _cleaning_tweet(text: str, **kwargs):
    reduce2len = kwargs.get('reduce2len', None)
    clean_punct = kwargs.get("clean_punct", False)
    keep_neutral_punct = kwargs.get("keep_neutral_punct", False)
    clean_num = kwargs.get("clean_num", False)
    replace_num_with = str(kwargs.get("replace_num_with", ""))
    text = cleaning_default(text)  # General preprocessing
    text = clean(text,
                 fix_unicode=True,  # fix various unicode errors
                 to_ascii=True,  # transliterate to closest ASCII representation
                 no_line_breaks=True,  # fully strip line breaks as opposed to only normalizing them
                 no_urls=True,  # replace all URLs with a special token
                 no_emails=True,  # replace all email addresses with a special token
                 no_phone_numbers=True,  # replace all phone numbers with a special token
                 no_numbers=clean_num,  # replace all numbers with a special token
                 no_digits=clean_num,  # replace all digits with a special token
                 no_currency_symbols=True,  # replace all currency symbols with a special token
                 no_punct=False,  # remove punctuations
                 replace_with_punct="",  # instead of removing punctuations you may replace them
                 replace_with_url="",
                 replace_with_email="",
                 replace_with_phone_number="",
                 replace_with_number=replace_num_with,
                 replace_with_digit=replace_num_with,
                 replace_with_currency_symbol="",
                 lang="en"  # set to 'de' for German special handling
                 )
    if clean_punct:
        text = _remove_punct(text, keep_neutral_punct)

    if reduce2len is not None:
        text = reduce_lengthening(text, reduce2len)
    return text


def cleaning_tweet(text_list: List[str], check_spell: bool = False, batch_size: int = 512,
                   is_test: bool = False, n_workers: int = -1, **kwargs) -> List[str]:
    """This function cleans (preprocess) sentences in text_list as if they are tweets

    Args:
        text_list (List[str]): list containing the strings texts of tweets to clean.
        check_spell (bool, optional): If any misspellings should be also corrected. Defaults to False.
        batch_size (int, optional): The texts in text_list is processed in batches of size batch_size.
            Only valid when check_spell is True. Defaults to 512.
        is_test (bool, optional): Whether the text_list corresponds to the testing data and not the training or validation data.
            Defaults to False.
        n_workers (int, optional): number of workers (=number of processes) to use when cleaning the tweets.
            Defaults to the number of logical processors.
        kwargs:
            reduce2len (int, optional): the minimum length of a tweet.
            clean_punct (bool, optional): If any punctuations should be removed. Defaults to False.
            keep_neutral_punct (bool, optional): If neutral punctuations should be kept during cleaning
                valid only when clean_punct is True
            clean_num (bool, optional): If any numbers and digits should be removed. Defaults to False.
            replace_num_with (str, optional): Replace numbers and digits with a given string
                valid only when clean_num is True
    Returns:
        List[str]: A list containing cleaned (preprocessed) strings of tweets
    """
    if type(text_list) is str:
        is_test = True
        check_spell = False
        text_list = [text_list]

    if is_test:
        _tmp = []
        for test_sent in text_list:
            _id = test_sent.split(',', 1)[0]
            _sent = test_sent.split(',', 1)[-1]
            _sent_cleaned = _cleaning_tweet(_sent, **kwargs)
            _result = ",".join([str(_id), _sent_cleaned])
            _tmp.append(_result)
        text_list = _tmp
    else:
        if n_workers == -1:
            n_workers = multiprocessing.cpu_count()
        logger.info(f"Cleaning text by {n_workers} workers. Please wait ...")
        _text_list = text_list
        tmp = Parallel(n_jobs=n_workers)(delayed(_cleaning_tweet)(tel, **kwargs) for tel in _text_list)
        text_list = tmp

    if check_spell is True:
        import neuspell
        from neuspell import BertChecker

        if Path().resolve().parts[1] == 'cluster':
            spell_checker_path = Path(os.getenv("SCRATCH"), '.cache', 'subwordbert-probwordnoise')
        else:
            spell_checker_path = Path(PROJECT_PATH, 'src', 'preprocessing', 'subwordbert-probwordnoise')
        spell_checker_exists = spell_checker_path.exists()
        if Path().resolve().parts[1] == 'cluster' and not spell_checker_exists:
            logger.info("Set the proxy for downloading spell checker")
            proxy = "http://proxy.ethz.ch:3128"
            os.environ['HTTP_PROXY'] = proxy
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
            for test_sent in tqdm(text_list):
                _id = test_sent.split(',', 1)[0]
                _sent = test_sent.split(',', 1)[-1]
                if len(_sent) > 0:
                    _sent_cleaned = checker.correct(_sent)
                else:
                    _sent_cleaned = ''
                results.append(','.join([_id, _sent_cleaned]))
        else:
            for i in tqdm(range(0, len(text_list), batch_size)):
                text_batch = text_list[i:i + batch_size]
                text_batch = list(filter(None, text_batch))
                text_batch = checker.correct_strings(text_batch)
                results.extend(text_batch)
                torch.cuda.empty_cache()
        text_list = results

    return text_list


def cleaningMap(clFunction: str) -> Callable:
    """ Maps the cleaning function names to functions

    Args:
        clFunction (str): function name in str. It can be one of default, masks, strip, tweet
    Returns:
        Callable: function which does the cleaning
    """
    d = {
        "default": cleaning_default,
        "strip": cleaning_strip,
        "tweet": cleaning_tweet,
    }
    assert clFunction in d.keys(), f"There is no {clFunction} in cleaning functions"
    return d[clFunction]


def main(args: List[str]):
    """The function for cleaning data and export the cleaned data to a new file"""
    argv = {a.split('=')[0]: a.split('=')[1] for a in args[1:]}
    data_path = argv.get('data_path', Path(DATA_PATH, 'test_data.txt'))
    reduce2len = argv.get('redcuce2len', None)
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
    input_data = list(filter(None, input_data))

    cleaned = cleaning_tweet(input_data, reduce2len=reduce2len,is_test=True if 'test' in input_file_name else False)
    cleaned_lines = [t + '\n' for t in cleaned]

    with open(Path(output_path), 'w') as fw:
        fw.writelines(cleaned_lines)


if __name__ == '__main__':
    if Path().resolve().parts[1] == 'cluster':
        os.environ["TRANSFORMERS_CACHE"] = os.path.join(os.getenv("SCRATCH"), '.cache/huggingface/')
    main(sys.argv)
