import re
from typing import Dict, Callable, Union

import pandas as pd


def cleaning_default(text: Union[str, list]):
    to_be_removed = r'(<.*?>)|[\'\"]'
    if type(text) is str:
        return re.sub(to_be_removed, '', text.strip())
    else:
        _tmp = pd.Series(text)
        _tmp = _tmp.str.strip()
        _result = _tmp.str.replace(to_be_removed, '').to_list()
        return _result


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
