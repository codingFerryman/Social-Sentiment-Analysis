import re
from typing import Dict, Callable, Union

import pandas as pd
from cleantext import clean


def cleaning_default(text: Union[str, list]):
    to_be_removed = r'(<.*?>)|[\'\"]'
    if type(text) is str:
        return re.sub(to_be_removed, '', text.strip())
    else:
        _tmp = pd.Series(text)
        _tmp = _tmp.str.strip()
        _result = _tmp.str.replace(to_be_removed, '').to_list()
        return _result


def cleaning_default_dev(text: Union[str, list]):
    # TODO: This part is still in development
    if type(text) is str:
        text = cleaning_masks(text)
        text = clean(text,
                     fix_unicode=True,  # fix various unicode errors
                     to_ascii=True,  # transliterate to closest ASCII representation
                     lower=True,  # lowercase text
                     no_line_breaks=True,  # fully strip line breaks as opposed to only normalizing them
                     no_urls=True,  # replace all URLs with a special token
                     no_emails=True,  # replace all email addresses with a special token
                     no_phone_numbers=True,  # replace all phone numbers with a special token
                     no_numbers=True,  # replace all numbers with a special token
                     no_digits=True,  # replace all digits with a special token
                     no_currency_symbols=True,  # replace all currency symbols with a special token
                     no_punct=True,  # remove punctuations
                     replace_with_punct=" ",  # instead of removing punctuations you may replace them
                     replace_with_url="",
                     replace_with_email="",
                     replace_with_phone_number="",
                     replace_with_number="",
                     replace_with_digit=" ",
                     replace_with_currency_symbol="",
                     lang="en"
                     )
        return text


def cleaning_masks(text: Union[str, list]):
    to_be_removed = r'(<.*?>)'
    if type(text) is str:
        return re.sub(to_be_removed, '', text.strip())
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


def cleaningMap() -> Dict[str, Callable]:
    return {
        "default": cleaning_default,
        "masks": cleaning_strip,
        "strip": cleaning_strip
    }
