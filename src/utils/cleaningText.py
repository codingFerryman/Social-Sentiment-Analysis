import re
from typing import Dict, Callable


def cleaning_default(text):
    return re.sub(r'(<.*?>)|[\'\"]', '', text.strip())


def cleaning_strip(text):
    return text.strip()


def cleaningMap() -> Dict[str, Callable]:
    return {
        "default": cleaning_default,
        "strip": cleaning_strip
    }
