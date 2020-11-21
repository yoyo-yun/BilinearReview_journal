# ======================================= #
# ------------ IMDB DataModel ----------- #
# ======================================= #
import torch
from torchtext.data import NestedField, Field, Dataset, Example
from torchtext.data.iterator import BucketIterator
from torchtext.vocab import Vectors
import pandas as pd
import pprint
pp = pprint.PrettyPrinter(indent=4)

from datasets.utils import clean_string, split_sents


class IMDB(Dataset):
    NAME = 'IMDB'
    NUM_CLASSES = 10
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

        pd_reader = pd.read_csv(path, header=None, skiprows=0, encoding="utf-8", sep='\t\t', engine='python')
        usrs = []
        products = []
        labels = []
        texts = []
        for i in range(len(pd_reader[0])):
            usrs.append(pd_reader[0][i])
            products.append(pd_reader[1][i])
            labels.append(pd_reader[2][i])
            texts.append(pd_reader[3][i])

        examples = [make_example([usr, product, label, text], fields) for usr, product, label, text in
                    zip(usrs, products, labels, texts)]

        if isinstance(fields, dict):
            fields, field_dict = [], fields
            for field in field_dict.values():
                if isinstance(field, list):
                    fields.extend(field)
                else:
                    fields.append(field)

        super(IMDB, self).__init__(examples, fields, **kwargs)

    @classmethod
    def splits(cls,
               path,
               train='imdb.train.txt.ss',
               validation='imdb.dev.txt.ss',
               test='imdb.test.txt.ss',
               **kwargs):
        return super(IMDB, cls).splits(
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
            cls.TEXT_FIELD.nesting_field.vocab = Vectors(name='imdb_text', cache=vectors_path)
        else:
            cls.TEXT_FIELD.vocab = Vectors(name='imdb_text', cache=vectors_path)
        cls.USR_FIELD.vocab = Vectors(name='imdb_usr', cache=vectors_path)
        cls.PRD_FIELD.vocab = Vectors(name='imdb_prd', cache=vectors_path)

        return BucketIterator.splits((train, val, test), batch_size=batch_size, repeat=False, shuffle=shuffle,
                                     sort_within_batch=True, device=device)


class IMDBHierarchical(IMDB):
    NESTING_FIELD = Field(batch_first=True, tokenize=clean_string)
    TEXT_FIELD = NestedField(NESTING_FIELD, tokenize=split_sents, include_lengths=True)
