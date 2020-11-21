# ======================================= #
# ------------ IMDB DataModel ----------- #
# ======================================= #
import torch
from torchtext.data import NestedField, Field, Dataset, Example
from torchtext.data.iterator import BucketIterator
from torchtext.vocab import Vectors
import pandas as pd
from collections import Counter
import math
import pprint
pp = pprint.PrettyPrinter(indent=4)

from datasets.utils import clean_string, split_sents, split_sents_nltk


class DIGITAL(Dataset):
    NAME = 'Digital_Music'
    NUM_CLASSES = 5
    IS_MULTILABEL = False

    TEXT_FIELD = Field(batch_first=True, tokenize=clean_string, include_lengths=True)
    USR_FIELD = Field(batch_first=True)
    PRD_FIELD = Field(batch_first=True)
    LABEL_FIELD = Field(sequential=False, use_vocab=False, batch_first=True, dtype=torch.long)

    @staticmethod
    def sort_key(ex):
        return len(ex.text)

    def __init__(self, path, fields, **kwargs):
        make_example = Example.fromlist

        pd_reader = pd.read_csv(path, header=None, skiprows=0, encoding="utf-8", sep='\t', engine='python')
        usrs = []
        products = []
        labels = []
        texts = []
        for i in range(len(pd_reader[0])):
            usrs.append(pd_reader[0][i])
            products.append(pd_reader[1][i])
            labels.append(pd_reader[2][i])
            texts.append(pd_reader[3][i])

        examples = []
        for usr, product, label, text in zip(usrs, products, labels, texts):
            try:
                example = make_example([usr, product, label, text], fields)
                # print(math.isnan(example.text))
                # print(example)
                examples.append(example)
                if math.isnan(example.text):
                    del examples[-1]
                    continue
            except:
                continue
        # print(examples)
        # exit()
        if isinstance(fields, dict):
            fields, field_dict = [], fields
            for field in field_dict.values():
                if isinstance(field, list):
                    fields.extend(field)
                else:
                    fields.append(field)

        super(DIGITAL, self).__init__(examples, fields, **kwargs)

    @classmethod
    def splits(cls,
               path,
               train='Digital_Music_train.txt',
               validation='Digital_Music_dev.txt',
               test='Digital_Music_test.txt',
               **kwargs):
        return super(DIGITAL, cls).splits(
            path, train=train, validation=validation, test=test,
            fields=[('usr', cls.USR_FIELD),
                    ('prd', cls.PRD_FIELD),
                    ('label', cls.LABEL_FIELD),
                    ('text', cls.TEXT_FIELD)])

    @classmethod
    def iters(cls, path, batch_size=64, shuffle=True, device=0, vectors_path=None):
        assert vectors_path is not None, print("should generate initial vectors first, by: "
                  "python get_save_vectors.py")
        train, val, test = cls.splits(path)
        if isinstance(cls.TEXT_FIELD, NestedField):
            cls.TEXT_FIELD.nesting_field.vocab = Vectors(name='digital_text_200', cache=vectors_path)
        else:
            cls.TEXT_FIELD.vocab = Vectors(name='digital_text', cache=vectors_path)
        cls.USR_FIELD.vocab = Vectors(name='digital_usr', cache=vectors_path)
        cls.PRD_FIELD.vocab = Vectors(name='digital_prd', cache=vectors_path)

        return BucketIterator.splits((train, val, test), batch_size=batch_size, repeat=False, shuffle=shuffle,
                                     sort_within_batch=True, device=device)


class DIGITALHierarchical(DIGITAL):
    NESTING_FIELD = Field(batch_first=True, tokenize=clean_string)
    TEXT_FIELD = NestedField(NESTING_FIELD, tokenize=split_sents_nltk, include_lengths=True)
