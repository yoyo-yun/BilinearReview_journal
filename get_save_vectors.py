import os
import sys
import json
import torch
import numpy as np
from cfgs.constants import DATASET_MAP, PRD_VECTORS_MAP, USR_VECTORS_MAP, TEXT_VECTORS_MAP, DATASET_PATH_MAP, \
    PRE_TRAINED_VECTOR_PATH, EXTRA_PRD_VECTORS_MAP
from datasets.utils import UnknownWordVecCache, UnknownUPVecCache
from datasets.utils import split_sents_nltk


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


def test_load_vectors(data_type):
    import time
    from collections import Counter
    start_time = time.time()
    print("====loading vectors...")
    train, val, test = DATASET_MAP[data_type].iters(path=DATASET_PATH_MAP[data_type], batch_size=64, shuffle=True,
                                                    device=0, vectors_path=PRE_TRAINED_VECTOR_PATH)
    print("done")
    print()
    end_time = time.time()


    words_per_setence = []
    sentences_per_document = []

    for batch in train:
        sentences_per_document.extend(batch.text[1])
        for j in batch.text[2]:
            words_per_setence.extend(j.tolist())
    for batch in val:
        sentences_per_document.extend(batch.text[1])
        for j in batch.text[2]:
            words_per_setence.extend(j.tolist())
    for batch in test:
        sentences_per_document.extend(batch.text[1])
        for j in batch.text[2]:
            words_per_setence.extend(j.tolist())

    new_words_per_setence = [i for i in words_per_setence if i > 0]
    new_sentences_per_document = [i for i in sentences_per_document if i > 0]
    print("average word in a sentecce     is:" + str(np.array(new_words_per_setence).mean()))
    print("average sentence in a document is:" + str(torch.stack(new_sentences_per_document).float().mean()))

    # import scipy.sparse as sp
    #
    # n_users = len(train.dataset.USR_FIELD.vocab.stoi)
    # n_items = len(train.dataset.PRD_FIELD.vocab.stoi)
    # print(n_users)
    # print(n_items)
    # usr_prd_map = sp.dok_matrix((n_users, n_items), dtype=np.float32)
    # for batch in train:
    #     users = batch.usr.squeeze(-1)
    #     items = batch.prd.squeeze(-1)
    #     for user, item in zip(users, items):
    #         usr_prd_map[user, item] = 1.
    # for batch in val:
    #     users = batch.usr.squeeze(-1)
    #     items = batch.prd.squeeze(-1)
    #     for user, item in zip(users, items):
    #         usr_prd_map[user, item] = 1.
    # for batch in test:
    #     users = batch.usr.squeeze(-1)
    #     items = batch.prd.squeeze(-1)
    #     for user, item in zip(users, items):
    #         usr_prd_map[user, item] = 1.
    # print(usr_prd_map.sum(1))
    # print(usr_prd_map.sum(1).mean())
    # print(usr_prd_map.sum(0))
    # print(usr_prd_map.sum(0).mean())


    # print(np.array(new_words_per_setence))
    # print(torch.stack(sentences_per_document))
    # print(torch.cat(words_per_setence, -1))
    # print(torch.cat(sentences_per_document, -1))

    # doc_length = Counter()
    # sent_length = Counter()
    #
    # for batch in train:
    #     doc_length.update(batch.text[1].tolist())
    #     for j in batch.text[2]:
    #         sent_length.update(j.tolist())
    # for batch in val:
    #     doc_length.update(batch.text[1].tolist())
    #     for j in batch.text[2]:
    #         sent_length.update(j.tolist())
    # for batch in test:
    #     doc_length.update(batch.text[1].tolist())
    #     for j in batch.text[2]:
    #         sent_length.update(j.tolist())

    # example an iterator
    # for i in train:
    #     # print("====text size...")
    #     # # text: (batch, sent, seq) document_length:(batch,) sentence_length: (batch, sent)
    #     # print("text:{}\tdoc_length:{}\tsent_length{}".format(i.text[0].shape, i.text[1].shape, i.text[2].shape))
    #     # # label: (batch,)
    #     # print("label size")
    #     # print("text:{}".format(i.label.shape))
    #     # # user: (batch, 1)
    #     # print("user size")
    #     # print("usr:{}".format(i.usr.shape))
    #     # # product: (batch, 1)
    #     # print("product size")
    #     # print("prd:{}".format(i.prd.shape))
    #     # print()
    #     #
    #     # print("====example...")
    #     # print("text:        ", str(i.text[0][0]))
    #     # print("doc_length:  ", str(i.text[1][0]))
    #     # print("sent_length: ", str(i.text[2][0]))
    #     # print("label:       ", str(i.label[0]))
    #     # print("user:        ", str(i.usr[0]))
    #     # print("product:     ", str(i.prd[0]))
    #     # break
    #
    #     # text: (batch, sent, seq) document_length:(batch,) sentence_length: (batch, sent)
    #     # for id, j in enumerate(i.text[1]):
    #     #     if j > 200:
    #     #         print("".center(60, '-'))
    #     #         print(j)
    #     #         for sent in i.text[0][id]:
    #     #             t = " ".join([DATASET_MAP[data_type].TEXT_FIELD.nesting_field.vocab.itos[token] if token!=1 else "" for token in sent]).strip()
    #     #             print(t)
    #                 # print(sent)
    #     # doc_length.append(i.text[1][0])
    #     # sent_length.extend(i.text[2][0])
    #     # for j in i.text[1]:
    #     #     # print(j)
    #     #     doc_length.update(j.item())
    #     doc_length.update(i.text[1].tolist())
    #     for j in i.text[2]:
    #         sent_length.update(j.tolist())
    # print(max(np.array(doc_length)))
    # print(max(np.array(sent_length)))
    # print(doc_length)
    # print(sent_length)
    print("taking times: {}s.".format(end_time - start_time))

    print(DATASET_MAP[data_type].TEXT_FIELD.nesting_field.vocab.vectors)


def generate_extra_embedding(embedding_file_path):
    stoi = {}
    index = 0
    word_embedding = []
    with open(embedding_file_path, 'r') as f:
        for line in f:
            check = line.strip().split()
            if len(check) == 2: continue
            line = line.strip().split()
            try:
                embedding = [float(s) for s in line[1:]]
            except:
                continue
            word_embedding.append(embedding)
            stoi[line[0]] = index
            index += 1

    # print(stoi)
    # print(word_embedding[:10].__str__())
    return stoi, torch.from_numpy(np.asarray(word_embedding))


def initial_vectors_from_pretrained(data_type):
    def save_vectors(path, field, dim):
        # self.itos, self.stoi, self.vectors, self.dim
        data = field.itos, field.stoi, field.vectors, dim
        torch.save(data, path)

    print("====loading draw data...")
    train, val, test = DATASET_MAP[data_type].splits(path=DATASET_PATH_MAP[data_type], unk_init=UnknownWordVecCache.unk)
    print("done")
    print()

    print("====building vectors...")
    DATASET_MAP[data_type].TEXT_FIELD.build_vocab(train, val, test, vectors=None)
    DATASET_MAP[data_type].USR_FIELD.build_vocab(train, val, test, vectors=None)
    DATASET_MAP[data_type].PRD_FIELD.build_vocab(train, val, test, vectors=None)

    text_filed = DATASET_MAP[data_type].TEXT_FIELD.vocab
    usr_filed = DATASET_MAP[data_type].USR_FIELD.vocab
    prd_filed = DATASET_MAP[data_type].PRD_FIELD.vocab

    text_stoi, text_vectores = generate_extra_embedding(EXTRA_PRD_VECTORS_MAP[data_type])
    train.TEXT_FIELD.nesting_field.vocab.set_vectors(text_stoi, text_vectores, 200, UnknownUPVecCache.unk)
    train.USR_FIELD.vocab.set_vectors({}, [], 100, UnknownUPVecCache.unk)
    train.PRD_FIELD.vocab.set_vectors({}, [], 100, UnknownUPVecCache.unk)

    print("done")
    print()

    print("====saving vectors...")
    save_vectors(TEXT_VECTORS_MAP[data_type], text_filed, 200)
    save_vectors(USR_VECTORS_MAP[data_type], usr_filed, 100)
    save_vectors(PRD_VECTORS_MAP[data_type], prd_filed, 100)
    print("done")


def test_graph_vectors(data_type):
    import time
    from torchtext.data.iterator import BucketIterator

    start_time = time.time()
    print("====loading draw data...")
    train, dev, test = DATASET_MAP[data_type].splits(path=DATASET_PATH_MAP[data_type], unk_init=UnknownWordVecCache.unk)
    print("done")
    print()

    print("====building vectors...")
    DATASET_MAP[data_type].TEXT_FIELD.build_vocab(train, dev, test, vectors='glove.840B.300d')
    DATASET_MAP[data_type].USR_FIELD.build_vocab(train, dev, test, vectors=None)
    DATASET_MAP[data_type].PRD_FIELD.build_vocab(train, dev, test, vectors=None)
    print("done")
    print()

    train, dev, test = BucketIterator.splits((train, dev, test), batch_size=64, repeat=False,
                                             sort_within_batch=True, device='cpu')

    # print(DATASET_MAP[data_type].TEXT_FIELD.vocab.itos)
    # print("4 is " + str(DATASET_MAP[data_type].TEXT_FIELD.vocab.itos[4]))
    # example an iterator
    for i in train:
        print("====text size...")
        # text: (batch, sent, seq) document_length:(batch,) sentence_length: (batch, sent)
        print("text:{}\tdoc_length:{}\tsent_length{}\tgraph_shape".format(i.text[0].shape, i.text[1].shape,
                                                                          i.text[2].shape, i.text[3].shape))
        # label: (batch,)
        print("label size")
        print("text:{}".format(i.label.shape))
        # user: (batch, 1)
        print("user size")
        print("usr:{}".format(i.usr.shape))
        # product: (batch, 1)
        print("product size")
        print("prd:{}".format(i.prd.shape))
        print()

        print("====example...")
        print("text:        ", str(i.text[0][0]))
        print("doc_length:  ", str(i.text[1][0]))
        print("sent_length: ", str(i.text[2][0]))
        print("graphs:      ", str(i.text[3][0]))
        print("label:       ", str(i.label[0]))
        print("user:        ", str(i.usr[0]))
        print("product:     ", str(i.prd[0]))
        break
    end_time = time.time()
    print("taking times: {}s.".format(end_time - start_time))


# dataset: Digital_Music, Industrial_and_Scientific, Software
def load_draw_data(dataset):
    import pandas as pd
    import gzip
    import re
    import numpy as np
    def parse(path):
        g = gzip.open(path, 'rb')
        for l in g:
            yield json.loads(l)

    def getDF(path):
        i = 0
        df = {}
        for d in parse(path):
            try:
                review = d["reviewText"]
            except:
                continue
            sents = split_sents_nltk(review.replace('\n', ""))
            lengths = [len(s.split()) for s in sents]
            try:
                if len(sents) > 100 or max(lengths) > 300:
                    continue
            except:
                continue
            df[i] = d
            i += 1
        return pd.DataFrame.from_dict(df, orient='index')

    data_path = 'corpus/raw_data/{}_5.json.gz'.format(dataset)
    savd_path_template = 'corpus/amazon/{}_{}.txt'
    df = getDF(data_path)
    df = df.sample(frac=1, random_state=np.random.RandomState(seed=1))
    total_example = df.shape[0]
    # print(total_example)
    train_length = int(total_example * 0.8)
    dev_length = int(total_example * 0.1)
    test_length = total_example - train_length - dev_length
    train_df = df[['reviewerID', 'asin', 'overall', 'reviewText']][:train_length]
    dev_df = df[['reviewerID', 'asin', 'overall', 'reviewText']][train_length:train_length + dev_length]
    test_df = df[['reviewerID', 'asin', 'overall', 'reviewText']][-test_length:]
    train_df.to_csv(savd_path_template.format(dataset, 'train'), sep='\t', header=None, index=None)
    dev_df.to_csv(savd_path_template.format(dataset, 'dev'), sep='\t', header=None, index=None)
    test_df.to_csv(savd_path_template.format(dataset, 'test'), sep='\t', header=None, index=None)


if __name__ == '__main__':
    # test_load_vectors('digital')
    # test_graph_vectors('imdb')
    # load_draw_data('Digital_Music')
    # load_draw_data('Industrial_and_Scientific')
    # load_draw_data('Software')
    # pass
    # initial_vectors_from_pretrained('yelp_13')
    # initial_vectors('digital')
    # initial_vectors('industrial')
    # initial_vectors('software')
    initial_vectors('imdb')
    # test_load_vectors('yelp_14')
    # initial_vectors_from_pretrained('imdb')
    # test_load_vectors('digital')
    # generate_extra_embedding('pretrained_Vectors/imdb-embedding-200d.txt')
    # generate_extra_embedding('pretrained_Vectors/yelp-2013-embedding-200d.txt')
    # generate_extra_embedding('pretrained_Vectors/yelp-2014-embedding-200d.txt')
    # data_type = 'imdb'
    # train, val, test = DATASET_MAP[data_type].splits(path=DATASET_PATH_MAP[data_type], unk_init=UnknownWordVecCache.unk)
    # DATASET_MAP[data_type].TEXT_FIELD.build_vocab(train, val, test, vectors='glove.840B.300d')
    # DATASET_MAP[data_type].USR_FIELD.build_vocab(train, val, test, vectors=None)
    # DATASET_MAP[data_type].PRD_FIELD.build_vocab(train, val, test, vectors=None)
    # text_filed = DATASET_MAP[data_type].TEXT_FIELD.vocab
    # usr_filed = DATASET_MAP[data_type].USR_FIELD.vocab
    # prd_filed = DATASET_MAP[data_type].PRD_FIELD.vocab
    #
    # train.USR_FIELD.vocab.set_vectors({}, [], 300, UnknownUPVecCache.unk)
    # train.PRD_FIELD.vocab.set_vectors({}, [], 300, UnknownUPVecCache.unk)
    # path = TEXT_VECTORS_MAP['imdb']
    # print(path)
    # itos, stoi, vectors, dim = torch.load(path)
    # print(itos[:10])
    # print(vectors[:10])
