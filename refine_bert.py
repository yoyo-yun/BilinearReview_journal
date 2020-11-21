import re
import os
import torch
import pandas as pd
import torch.utils.data
from torch.optim import Adam
from collections import Counter
from torch.utils.data import DataLoader
import numpy as np

class InputExample(object):
    def __init__(self, guid=None, text=None, user=None, product=None, label=None):
        self.guid = guid
        self.text = text
        self.label = label
        self.user = user
        self.product = product

class Data(torch.utils.data.Dataset):
    sort_key = None
    def __init__(self, *data):
        assert all(len(data[0]) == len(d) for d in data)
        self.data = data

    def __len__(self):
        return len(self.data[0])

    def __getitem__(self, index):
        return tuple(d[index] for d in self.data)

class SentenceProcessor(object):
    NAME = 'SENTENCE'

    def get_sentences(self):
        raise NotImplementedError

    def _create_examples(self, documents, type):
        examples = []
        for (i, line) in enumerate(documents):
            guid = "%s-%s" % (type, i)
            try:
                text = clean_document(line[2])
            except:
                continue
            examples.append(
                InputExample(guid=guid, user=line[0], product=line[1], text=text, label=int(line[3]) - 1))
        return examples

    def _read_file(self, dataset):
        pd_reader = pd.read_csv(dataset, header=None, skiprows=0, encoding="utf-8", sep='\t', engine='python')
        documents = []
        for i in range(len(pd_reader[0])):
            # print(pd_reader[3][i])
            # if math.isnan(pd_reader[3][i]):
            #     continue
            document = list([pd_reader[0][i], pd_reader[1][i], pd_reader[3][i], pd_reader[2][i]])
            documents.append(document)
        return documents

    def _get_attributes(self, *datasets):
        users = Counter()
        products = Counter()
        ATTR_MAP = {
            'user': int(0),
            'product': int(1)
        }
        for dataset in datasets:
            for document in dataset:
                users.update([document[ATTR_MAP["user"]]])
                products.update([document[ATTR_MAP["product"]]])
        return tuple([users, products])

def clean_document(document):
    string = re.sub(r"\n{2,}", " ", document)
    string = re.sub(r"[^A-Za-z0-9(),!?\'.`]", " ", string)
    return string.lower().strip()

class Digital(SentenceProcessor):
    NAME = 'digital'
    NUM_CLASSES = 10

    def __init__(self, data_dir='corpus'):
        self.d_train = self._read_file(os.path.join(data_dir, 'amazon', 'Digital_Music_train.txt'))
        self.d_dev = self._read_file(os.path.join(data_dir, 'amazon', 'Digital_Music_dev.txt'))
        self.d_test = self._read_file(os.path.join(data_dir, 'amazon', 'Digital_Music_test.txt'))

    def get_documents(self):
        train = self._create_examples(self.d_train, 'train')
        dev = self._create_examples(self.d_dev, 'dev')
        test = self._create_examples(self.d_test, 'test')
        return tuple([train, dev, test])

    def get_attributes(self):
        return self._get_attributes(self.d_train, self.d_dev,
                                    self.d_test)  # tuple(attributes) rather tuple(users, products)

class Industrial(SentenceProcessor):
    NAME = 'industrial'
    NUM_CLASSES = 10

    def __init__(self, data_dir='corpus'):
        self.d_train = self._read_file(os.path.join(data_dir, 'amazon', 'Industrial_and_Scientific_train.txt'))
        self.d_dev = self._read_file(os.path.join(data_dir, 'amazon', 'Industrial_and_Scientific_dev.txt'))
        self.d_test = self._read_file(os.path.join(data_dir, 'amazon', 'Industrial_and_Scientific_test.txt'))

    def get_documents(self):
        train = self._create_examples(self.d_train, 'train')
        dev = self._create_examples(self.d_dev, 'dev')
        test = self._create_examples(self.d_test, 'test')
        return tuple([train, dev, test])

    def get_attributes(self):
        return self._get_attributes(self.d_train, self.d_dev,
                                    self.d_test)  # tuple(attributes) rather tuple(users, products)

class Software(SentenceProcessor):
    NAME = 'industrial'
    NUM_CLASSES = 10

    def __init__(self, data_dir='corpus'):
        self.d_train = self._read_file(os.path.join(data_dir, 'amazon', 'Software_train.txt'))
        self.d_dev = self._read_file(os.path.join(data_dir, 'amazon', 'Software_dev.txt'))
        self.d_test = self._read_file(os.path.join(data_dir, 'amazon', 'Software_test.txt'))

    def get_documents(self):
        train = self._create_examples(self.d_train, 'train')
        dev = self._create_examples(self.d_dev, 'dev')
        test = self._create_examples(self.d_test, 'test')
        return tuple([train, dev, test])

    def get_attributes(self):
        return self._get_attributes(self.d_train, self.d_dev,
                                    self.d_test)  # tuple(attributes) rather tuple(users, products)

DATASET = {
    'industrial': Industrial,
    'software': Software,
    'digital': Digital
}

def multi_acc(y, preds):
    preds = torch.argmax(torch.softmax(preds, dim=-1), dim=1)
    correct = torch.eq(preds, y).float()
    acc = correct.sum() / len(correct)
    return acc

def refine_bert(data):
    device = "cuda:1"
    loss_f = torch.nn.CrossEntropyLoss()
    dataset = DATASET[data]()
    s_train, s_dev, s_test = dataset.get_documents()
    # # training data
    train_text, train_label = [example.text for example in s_train], [example.label for example in s_train]
    train_dataset = Data(train_text, train_label)
    train_dataloader = DataLoader(train_dataset, batch_size=6, shuffle=True)

    # #dev data
    test_text, test_label = [example.text for example in s_test], [example.label for example in s_test]
    test_dataset = Data(test_text, test_label)
    test_dataloader = DataLoader(test_dataset, batch_size=6, shuffle=False)

    from transformers import BertTokenizer, BertConfig, BertForSequenceClassification

    pretrained_weights = 'bert-base-uncased'
    tokenizer = BertTokenizer.from_pretrained(pretrained_weights)
    model = BertForSequenceClassification.from_pretrained(pretrained_weights, num_labels=5).to(device)
    print(model.config)

    lr = 5e-5
    max_grad_norm = 1.0
    optimizer = Adam(model.parameters(), lr=lr)

    # training phrase
    for epoch in range(10):
        print("epoch: {:2}".format(epoch).center(60, "-"))
        lenght_data_loader = len(train_dataloader)
        acc = []
        for bs_id, batch in enumerate(train_dataloader):
            text, label = batch
            t = tokenizer.batch_encode_plus(text, padding='max_length',
                                            max_length=510,
                                            truncation=True,
                                            )
            input_ids = torch.tensor(t["input_ids"]).to(device)
            attention_mask = torch.tensor(t["attention_mask"]).to(device)
            labels = label.long().to(device)
            # print(input_ids.shape)
            # print(attention_mask.shape)
            model.train()
            logits = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
            )[0]
            # print(input_ids)
            loss = loss_f(logits, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(),
                                           max_grad_norm)  # Gradient clipping is not in AdamW anymore (so you can use amp without issue)
            optimizer.step()
            # scheduler.step()
            optimizer.zero_grad()
            print("\rIteration: {:3}/{:4} -- loss: {:^.3} acc: {:^.3}".format(bs_id, lenght_data_loader, loss,
                                                                              multi_acc(labels, logits)), end="")
            acc.append(multi_acc(labels, logits).item())
        print("\r{:<10} phrase===final acc: {:^.3}".format('training', np.array(acc).mean()) + "".center(30, " "),
              end='\n')

        # evlauation phrase
        lenght_data_loader = len(test_dataloader)
        acc = []
        for bs_id, batch in enumerate(test_dataloader):
            text, label = batch
            t = tokenizer.batch_encode_plus(text, padding='max_length',
                                            max_length=510,
                                            truncation=True,
                                            )
            input_ids = torch.tensor(t["input_ids"]).to(device)
            attention_mask = torch.tensor(t["attention_mask"]).to(device)
            labels = label.long().to(device)
            model.eval()
            logits = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
            )[0]
            # print(input_ids)
            loss = loss_f(logits, labels)
            print("\rIteration: {:3}/{:4} -- loss: {:^.3} acc: {:^.3}".format(bs_id, lenght_data_loader, loss,
                                                                              multi_acc(labels, logits)), end="")
            acc.append(multi_acc(labels, logits).item())
        print("\r{:<10} phrase===final acc: {:^.3}".format('evaluation', np.array(acc).mean()) + "".center(30, " "),
              end='\n')

if __name__ == "__main__":
    refine_bert("digital")





