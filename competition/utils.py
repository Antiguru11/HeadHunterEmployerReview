import re
import string
import unicodedata
from importlib import import_module

import nltk
import pymorphy2
from nltk.corpus import stopwords


_ru_morph = pymorphy2.MorphAnalyzer()


try:
    ru_stop_words = stopwords.words('russian')
    en_stop_words = stopwords.words('english')
    stop_words = set(ru_stop_words).union(set(en_stop_words))
except LookupError:
    if not nltk.download('stopwords'):
        raise RuntimeError("Couldn't load stopwords from nltk_data")
    ru_stop_words = stopwords.words('russian')
    en_stop_words = stopwords.words('english')
    stop_words = set(ru_stop_words).union(set(en_stop_words))


def preproc_str(input_str: str) -> str:
    output_str = input_str.lower()

    table = str.maketrans(string.punctuation, ' '*len(string.punctuation))
    output_str = output_str.translate(table)

    output_str = ''.join(list(filter(lambda x: not x.isnumeric(), output_str)))

    output_str = output_str.strip()

    output_str = unicodedata.normalize('NFKD', output_str)

    return output_str


def tfidf_preprocess_str(input_str: str) -> str:
    return re.sub('[^а-яА-Я ]+', '', preproc_str(input_str))


def tokenize(input_str: str) -> list[str]:
    global stop_words
    tokens = [t for t in input_str.split() if t not in stop_words]
    return tokens


def ru_normalize(input_str: str) -> str:
    global _ru_morph
    return _ru_morph.parse(input_str)[0].normal_form


def camel2snake(name: str) -> str:
    return re.sub(r'(?<!^)(?=[A-Z]{1}[a-z]+)', '_', name).lower()


def get_object(cls_name: str):
    parts = cls_name.split('.')

    module_name = '.'.join(parts[:-1])
    class_name = parts[-1]

    module = import_module(module_name)
    obj = getattr(module, class_name)
    return obj

