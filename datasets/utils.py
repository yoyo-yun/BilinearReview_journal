import json
import re
import sys
import csv
import torch

import numpy as np

csv.field_size_limit(sys.maxsize)


class UnknownWordVecCache(object):
    """
    Caches the first randomly generated word vector for a certain size to make it is reused.
    """
    cache = {}

    @classmethod
    def unk(cls, tensor):
        size_tup = tuple(tensor.size())
        if size_tup not in cls.cache:
            cls.cache[size_tup] = torch.Tensor(tensor.size())
            cls.cache[size_tup].uniform_(-0.25, 0.25)
        return cls.cache[size_tup]


class UnknownUPVecCache(object):
    @classmethod
    def unk(cls, tensor):
        return  tensor.uniform_(-0.25, 0.25)


def clean_string(string):
    string = re.sub(r"[^A-Za-z0-9(),!?\'`]", " ", string)
    string = re.sub(r"sssss", "", string)
    string = re.sub(r"\s{2,}", " ", string)
    return string.lower().strip().split()


def split_sents(string):
    string = re.sub(r"[!?]"," ", string)
    string = re.sub(r"\.{2,}", " ", string)
    return string.strip().split('.')

def clean_string_nltk(string):
    # string = re.sub(r"[^A-Za-z0-9(),!?\'`]", " ", string)
    return string.lower().strip().split()

def split_sents_nltk(string):
    from nltk.tokenize import sent_tokenize
    string = re.sub(r"[^A-Za-z().,!?\'`]", " ", string)
    string = re.sub(r"\n{2,}", " ", string)
    string = re.sub(r"\.{2,}", " ", string)
    return sent_tokenize(string.replace('\n',''))

#
# def clean_string(string):
#     string = re.sub(r"[^A-Za-z0-9(),.!?\'`]", " ", string)
#     string = re.sub(r"\s{2,}", " ", string)
#     return string.lower().strip().split()
#
# def split_sents(string):
#     string = re.sub(r"[!?.]", " ", string)
#     return string.strip().split('<sssss>')


def generate_ngrams(tokens, n=2):
    n_grams = zip(*[tokens[i:] for i in range(n)])
    tokens.extend(['-'.join(x) for x in n_grams])
    return tokens


def load_json(string):
    split_val = json.loads(string)
    return np.asarray(split_val, dtype=np.float32)


def process_labels(string):
    """
    Returns the label string as a list of integers
    """
    return [float(x) for x in string]