import re
import string
from importlib import import_module

import nltk
from nltk.corpus import stopwords


try:
    stop_words = set(stopwords.words('russian')).union(set(stopwords.words('english')))
except LookupError:
    if not nltk.download('stopwords'):
        raise RuntimeError("Couldn't load stopwords from nltk_data")
    stop_words = set(stopwords.words('russian')).union(set(stopwords.words('english')))


def preproc_str(input_str: str) -> str:
    output_str = input_str.lower()

    table = str.maketrans(string.punctuation, ' '*len(string.punctuation))
    output_str = output_str.translate(table)

    output_str = ''.join(list(filter(lambda x: not x.isnumeric(), output_str)))

    output_str = output_str.strip()

    return output_str


def tokenize(input_str: str) -> list[str]:
    global stop_words
    tokens = [t for t in input_str.split() if t not in stop_words]
    return tokens


def camel2snake(name: str) -> str:
    return re.sub(r'(?<!^)(?=[A-Z]{1}[a-z]+)', '_', name).lower()


def get_object(cls_name: str):
    parts = cls_name.split('.')

    module_name = '.'.join(parts[:-1])
    class_name = parts[-1]

    module = import_module(module_name)
    obj = getattr(module, class_name)
    return obj

