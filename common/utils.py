import torch
import torch.nn.functional as F
from cfgs.constants import DATASET_MAP, DATASET_PATH_MAP, PRE_TRAINED_VECTOR_PATH, DATASET_MAP_LSTM
from datasets.get_save_vectors import initial_vectors

class MyVector(object):
    def __init__(self):
        self.itos = None
        self.stoi = None
        self.vectors = None
        self.dim = None


def load_vectors(config, show_example=False, show_statics=False):
    import time
    start_time = time.time()
    print("====loading vectors...")
    try:
        train, val, test = DATASET_MAP[config.dataset].iters(path=DATASET_PATH_MAP[config.dataset],
                                                             batch_size=config.TRAIN.batch_size, shuffle=True,
                                                             device=config.device, vectors_path=PRE_TRAINED_VECTOR_PATH)
    except:
        print("Without local pre-loaded vectors. Now, loading from raw data...", end='\n'*2)
        initial_vectors(config.dataset)
        train, val, test = DATASET_MAP[config.dataset].iters(path=DATASET_PATH_MAP[config.dataset],
                                                             batch_size=config.TRAIN.batch_size, shuffle=True,
                                                             device=config.device, vectors_path=PRE_TRAINED_VECTOR_PATH)

    print("done!")
    print()
    end_time = time.time()

    # example an iterator
    if show_example:
        for i in train:
            print("====text size...")
            # text: (batch, sent, seq) length:(batch, sent)
            print("text:{}\tlength:{}".format(i.text[0].shape, i.text[1].shape))
            # label: (batch,)
            print("label size")
            print("text:{}".format(i.label.shape))
            # user: (batch, 1)
            print("user size")
            print("usr:{}".format(i.usr.shape))
            # product: (batch, 1)
            print("product size")
            print("prd:{}".format(i.prd.shape))

            print("====example...")
            print("text:    ", str(i.text[0][0]))
            print("length:  ", str(i.text[1][0]))
            print("label:   ", str(i.label[0]))
            print("user:    ", str(i.usr[0]))
            print("product: ", str(i.prd[0]))
            break
    print("taking times: {:.2f}s.".format(end_time - start_time))
    print()

    # statistic
    if show_statics:
        words_per_setence = []
        sentences_per_document = []
        for batch in train:
            sentences_per_document.append(batch.text[1])
            words_per_setence.append(batch.text[2])
        for batch in val:
            sentences_per_document.append(batch.text[1])
            words_per_setence.append(batch.text[2])
        for batch in test:
            sentences_per_document.append(batch.text[1])
            words_per_setence.append(batch.text[2])

    # update configuration
    config.num_classes = train.dataset.NUM_CLASSES
    config.text_embedding = train.dataset.TEXT_FIELD.nesting_field.vocab.vectors
    config.usr_embedding = train.dataset.USR_FIELD.vocab.vectors
    config.prd_embedding = train.dataset.PRD_FIELD.vocab.vectors
    config.pad_idx = train.dataset.TEXT_FIELD.nesting_field.vocab.stoi[train.dataset.TEXT_FIELD.pad_token]

    print("===Train size       : " + str(len(train.dataset)))
    print()
    print("===Validation size  : " + str(len(val.dataset)))
    print()
    print("===Test size        : " + str(len(test.dataset)))
    print()
    print("===common datasets information...")
    print("num_labels          : " + str(config.num_classes))
    print("pad_idx             : " + str(config.pad_idx))
    print("text vocabulary size: " + str(config.text_embedding.shape[0]))
    print("usr  vocabulary size: " + str(config.usr_embedding.shape[0]))
    print("prd  vocabulary size: " + str(config.prd_embedding.shape[0]))
    print()


    return train, val, test

def load_vectors_LSTM(config, show_example=False, show_statics=False):
    import time
    start_time = time.time()
    print("====loading vectors...")
    train, val, test = DATASET_MAP_LSTM[config.dataset].iters(path=DATASET_PATH_MAP[config.dataset],
                                                         batch_size=config.TRAIN.batch_size, shuffle=True,
                                                         device=config.device, vectors_path=PRE_TRAINED_VECTOR_PATH)
    print("done!")
    print()
    end_time = time.time()

    # example an iterator
    if show_example:
        for i in train:
            print("====text size...")
            # text: (batch, seq) length:(batch, sent)
            print("text:{}\tlength:{}".format(i.text[0].shape, i.text[1].shape))
            # label: (batch,)
            print("label size")
            print("text:{}".format(i.label.shape))
            # user: (batch, 1)
            print("user size")
            print("usr:{}".format(i.usr.shape))
            # product: (batch, 1)
            print("product size")
            print("prd:{}".format(i.prd.shape))

            print("====example...")
            print("text:    ", str(i.text[0][0]))
            print("length:  ", str(i.text[1][0]))
            print("label:   ", str(i.label[0]))
            print("user:    ", str(i.usr[0]))
            print("product: ", str(i.prd[0]))
            break
    print("taking times: {:.2f}s.".format(end_time - start_time))
    print()

    # statistic
    if show_statics:
        words_per_setence = []
        for batch in train:
            words_per_setence.append(batch.text[1])
        for batch in val:
            words_per_setence.append(batch.text[1])
        for batch in test:
            words_per_setence.append(batch.text[1])

    # update configuration
    config.num_classes = train.dataset.NUM_CLASSES
    config.text_embedding = train.dataset.TEXT_FIELD.vocab.vectors
    config.usr_embedding = train.dataset.USR_FIELD.vocab.vectors
    config.prd_embedding = train.dataset.PRD_FIELD.vocab.vectors
    config.pad_idx = train.dataset.TEXT_FIELD.vocab.stoi[train.dataset.TEXT_FIELD.pad_token]

    print("===Train size       : " + str(len(train.dataset)))
    print()
    print("===Validation size  : " + str(len(val.dataset)))
    print()
    print("===Test size        : " + str(len(test.dataset)))
    print()
    print("===common datasets information...")
    print("num_labels          : " + str(config.num_classes))
    print("pad_idx             : " + str(config.pad_idx))
    print("text vocabulary size: " + str(config.text_embedding.shape[0]))
    print("usr  vocabulary size: " + str(config.usr_embedding.shape[0]))
    print("prd  vocabulary size: " + str(config.prd_embedding.shape[0]))
    print()


    return train, val, test


def multi_acc(y, preds):
    """
    get accuracy

    preds: logits
    y: true label
    """
    preds = torch.argmax(F.softmax(preds), dim=1)
    correct = torch.eq(preds, y).float()
    acc = correct.sum() / len(correct)
    return acc

def multi_mse(y, preds):
    mse_loss = torch.nn.MSELoss()
    preds = torch.argmax(F.softmax(preds), dim=1)
    return mse_loss(y.float(), preds.float())
