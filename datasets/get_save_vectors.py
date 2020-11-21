import os
import sys
sys.path.append("..")
print(sys.path)

import torch
from cfgs.constants import DATASET_MAP, PRD_VECTORS_MAP, USR_VECTORS_MAP, TEXT_VECTORS_MAP, DATASET_PATH_MAP
from datasets.utils import UnknownWordVecCache, UnknownUPVecCache

# def initial_vectors(data_type):
#     def save_vectors(path, field, dim):
#     # self.itos, self.stoi, self.vectors, self.dim
#         data = field.itos, field.stoi, field.vectors, dim
#         torch.save(data, path)
#     train, val, test = DATASET_MAP[data_type].splits(path='corpus/imdb', unk_init=UnknownWordVecCache.unk)
#     text_filed = DATASET_MAP[data_type].TEXT_FIELD.build_vocab(train, val, test, vectors='glove.840B.300d')
#     usr_filed = DATASET_MAP[data_type].USR_FIELD.build_vocab(train, val, test, vectors=None)
#     prd_filed = DATASET_MAP[data_type].PRD_FIELD.build_vocab(train, val, test, vectors=None)
#
#     train.USR_FIELD.vocab.set_vectors({}, [], 300, UnknownUPVecCache.unk)
#     train.PRD_FIELD.vocab.set_vectors({}, [], 300, UnknownUPVecCache.unk)
#
#     save_vectors(TEXT_VECTORS_MAP[data_type], text_filed, 300)
#     save_vectors(USR_VECTORS_MAP[data_type], usr_filed, 100)
#     save_vectors(PRD_VECTORS_MAP[data_type], prd_filed, 100)


def initial_vectors(data_type):
    def save_vectors(path, field, dim):
        # self.itos, self.stoi, self.vectors, self.dim
        data = field.itos, field.stoi, field.vectors, dim
        torch.save(data, path)

    print("====loading draw data...")
    train, val, test = DATASET_MAP[data_type].splits(path=DATASET_PATH_MAP[data_type], unk_init=UnknownWordVecCache.unk)
    print("done")
    print()

    print("====building vectors...")
    DATASET_MAP[data_type].TEXT_FIELD.build_vocab(train, val, test, vectors='glove.840B.300d')
    # DATASET_MAP[data_type].TEXT_FIELD.build_vocab(train, val, test, vectors='glove.6B.50d')
    # DATASET_MAP[data_type].TEXT_FIELD.build_vocab(train, val, test, vectors='glove.6B.100d')
    # DATASET_MAP[data_type].TEXT_FIELD.build_vocab(train, val, test, vectors='glove.6B.200d')
    DATASET_MAP[data_type].USR_FIELD.build_vocab(train, val, test, vectors=None)
    DATASET_MAP[data_type].PRD_FIELD.build_vocab(train, val, test, vectors=None)
    print("done")
    print()
    text_filed = DATASET_MAP[data_type].TEXT_FIELD.vocab
    usr_filed = DATASET_MAP[data_type].USR_FIELD.vocab
    prd_filed = DATASET_MAP[data_type].PRD_FIELD.vocab

    train.USR_FIELD.vocab.set_vectors({}, [], 100, UnknownUPVecCache.unk)
    train.PRD_FIELD.vocab.set_vectors({}, [], 100, UnknownUPVecCache.unk)

    print("====saving vectors...")
    save_vectors(TEXT_VECTORS_MAP[data_type], text_filed, 300)
    save_vectors(USR_VECTORS_MAP[data_type], usr_filed, 100)
    save_vectors(PRD_VECTORS_MAP[data_type], prd_filed, 100)
    print("done")


if __name__ == '__main__':
    initial_vectors('imdb')