import os
import time
import torch
import datetime
import numpy as np
from functools import reduce
from models_v2.get_optim import get_Adam_optim, get_optim
from models_v2.model import Net
from common.utils import load_vectors, multi_acc, multi_mse
from transformers import BertTokenizer, BertModel


class Trainer(object):
    def __init__(self, config):
        self.config = config
        self.train_itr, self.dev_itr, self.test_itr = load_vectors(config)

        net = Net(self.config).to(self.config.device)
        if self.config.n_gpu > 1:
            self.net = torch.nn.DataParallel(net)
        else:
            self.net = net
        self.optim = get_Adam_optim(config, self.net)
        # self.optim = get_optim(config, self.net, len(self.train_itr.dataset))

        pretrained_weights = 'bert-base-uncased'
        self.tokenizer = BertTokenizer.from_pretrained(pretrained_weights)
        self.bert = BertModel.from_pretrained(pretrained_weights).to(self.config.device)

        self.early_stop = config.TRAIN.early_stop
        self.best_dev_acc = 0
        self.unimproved_iters = 0
        self.iters_not_improved = 0

    def train(self):
        # Save log information
        logfile = open(
            self.config.log_path +
            '/log_run_' + self.config.dataset + '_' + self.config.version + '.txt',
            'a+'
        )
        logfile.write(
            'nowTime: ' +
            datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S') +
            '\n' +
            'seed:' + str(self.config.seed) +
            '\n'
        )
        logfile.close()
        for epoch in range(0, self.config.TRAIN.max_epoch):
            self.net.train()
            train_loss, train_acc, train_rmse = self.train_epoch(epoch)

            logs = ("    Epoch:{:>2}    ".format(epoch)).center(88, "-") + "".center(70, " ") + '\n' + \
                   self.get_logging(train_loss, train_acc, train_rmse, eval="training")
            print("\r" + logs)

            # logging training logs
            self.logging(self.config.log_path + '/log_run_' + self.config.dataset + '_' + self.config.version + '.txt',
                         logs)

            # saving state
            state = {
                'state_dict': self.net.state_dict(),
                'usr_vectors': {
                    'stoi': self.train_itr.dataset.USR_FIELD.vocab.stoi,
                    'itos': self.train_itr.dataset.USR_FIELD.vocab.itos,
                    'dim': self.train_itr.dataset.USR_FIELD.vocab.dim
                },
                'prd_vectors': {
                    'stoi': self.train_itr.dataset.PRD_FIELD.vocab.stoi,
                    'itos': self.train_itr.dataset.PRD_FIELD.vocab.itos,
                    'dim': self.train_itr.dataset.PRD_FIELD.vocab.dim
                },
                'text_vectors': {
                    'stoi': self.train_itr.dataset.TEXT_FIELD.nesting_field.vocab.stoi,
                    'itos': self.train_itr.dataset.TEXT_FIELD.nesting_field.vocab.itos,
                    'dim': self.train_itr.dataset.TEXT_FIELD.nesting_field.vocab.dim
                }
            }

            self.net.eval()
            eval_loss, eval_acc, eval_rmse = self.eval(self.test_itr, state=state['state_dict'])
            eval_logs = self.get_logging(eval_loss, eval_acc, eval_rmse, eval="evaluating")
            print("\r" + eval_logs)

            # logging evaluating logs
            self.logging(self.config.log_path + '/log_run_' + self.config.dataset + '_' + self.config.version + '.txt',
                         eval_logs)

            # early stopping
            if eval_acc[eval_acc.argmax()] > self.best_dev_acc:
                self.unimproved_iters = 0
                self.best_dev_acc = eval_acc[eval_acc.argmax()]
                # saving models
                torch.save(
                    state,
                    self.config.ckpts_path + '/ckpt_{}_{}.pkl'.format(self.config.dataset, self.config.version)
                )

            else:
                self.unimproved_iters += 1
                if self.unimproved_iters >= self.config.TRAIN.patience and self.early_stop == True:
                    early_stop_logs = self.config.log_path + '/log_run_' + self.config.dataset + '_' + self.config.version + '.txt' + "\n" + \
                                      "Early Stopping. Epoch: {}, Best Dev Acc: {}".format(epoch, self.best_dev_acc)
                    print(early_stop_logs)
                    self.logging(
                        self.config.log_path + '/log_run_' + self.config.dataset + '_' + self.config.version + '.txt',
                        early_stop_logs)
                    break

    def train_epoch(self, epoch):
        loss_fn = torch.nn.CrossEntropyLoss()
        acc_fn = multi_acc
        mse_fn = multi_mse
        total_loss = []
        total_acc = []
        total_mse = []
        for step, batch in enumerate(self.train_itr):
            start_time = time.time()
            self.optim.zero_grad()
            text, doc_length, sent_length = batch.text
            new_text = []
            # text (bs, sentence, word)
            for document in text:
                new_document = []
                for sentence in document:
                    new_document.append(
                        " ".join([self.train_itr.dataset.TEXT_FIELD.nesting_field.vocab.itos[word] for word in
                                  sentence]).replace("<pad>", "").strip())
                new_text.append(new_document)

            ts = [self.tokenizer.batch_encode_plus(i, padding='longest', max_length=512, truncation=True) for i in
                  new_text]
            unpacked_input_ids = [t.input_ids for t in ts]
            unpacked_attention_mask = [t.attention_mask for t in ts]
            length_sentence = [len(t.input_ids) for t in ts]
            length_word = [len(t.input_ids[0]) for t in ts]
            max_length_sentence = max(length_sentence)
            max_length_word = max(length_word)

            pad_sentence_template = [[[0] * max_length_word] * (max_length_sentence - l) for l in length_sentence]
            pad_word_template = [[0] * (max_length_word - l) for l in length_word]

            for i in range(len(unpacked_input_ids)):
                for j in range(len(unpacked_input_ids[i])):
                    unpacked_input_ids[i][j] += pad_word_template[i]
                    unpacked_attention_mask[i][j] += pad_word_template[i]

            for i in range(len(unpacked_input_ids)):
                unpacked_input_ids[i] += pad_sentence_template[i]
                unpacked_attention_mask[i] += pad_sentence_template[i]

            packed_input_ids = torch.Tensor(unpacked_input_ids).long().to(self.config.device)
            packed_attention_mask = torch.Tensor(unpacked_attention_mask).long().to(self.config.device)
            packed_embedding = []
            with torch.no_grad():
                for input_ids, attention_mask in zip(packed_input_ids, packed_attention_mask):
                    packed_embedding.append(self.bert(input_ids, attention_mask, output_hidden_states=True)[2][-2])
            packed_embedding = torch.stack(packed_embedding, 0)
            label = batch.label - 1
            usr = batch.usr
            prd = batch.prd
            # logits : logits_bup, logits_text, logits_prd, logits_usr, logits_all
            # atts : word_text_atts, word_usr_atts, word_prd_atts, sent_text_atts, sent_usr_atts, sent_prd_atts
            logits, atts = self.net(packed_embedding, usr, prd, mask=packed_attention_mask)
            if self.config.inter_sent_level:
                weights = self.config.TRAIN.alpha_bup, 0, 0, 0, self.config.TRAIN.alpha_all
            else:
                weights = self.config.TRAIN.alpha_bup, self.config.TRAIN.alpha_text, self.config.TRAIN.alpha_prd, self.config.TRAIN.alpha_usr, self.config.TRAIN.alpha_all
            loss = [loss_fn(logit, label) for logit in logits]
            metric_acc = [acc_fn(label, logit) for logit in logits]
            metric_mse = [mse_fn(label, logit) for logit in logits]
            ultimate_loss = [l * w for l, w in zip(loss, weights)]
            ultimate_loss = reduce(lambda x, y: x + y, ultimate_loss)
            ultimate_loss.backward()
            self.optim.step()

            total_loss.append([ultimate_loss.data.cpu().numpy()] + [l.data.cpu().numpy() for l in loss])
            total_acc.append([m.data.cpu().numpy() for m in metric_acc])
            total_mse.append([m.data.cpu().numpy() for m in metric_mse])

            # monitoring results on every steps
            end_time = time.time()
            span_time = (end_time - start_time) * (
                    int(len(self.train_itr.dataset) / self.config.TRAIN.batch_size) - step)
            h = span_time // (60 * 60)
            m = (span_time % (60 * 60)) // 60
            s = (span_time % (60 * 60)) % 60 // 1
            print(
                "\rIteration: {:>4}/{} ({:>4.1f}%)  Loss: {:.5f} loss_bup: {:.5f} loss_text: {:.5f} loss_prd: {:.5f} loss_usr: {:.5f} loss_all: {:.5f} -ETA {:>2}h-{:>2}m-{:>2}s".format(
                    step, int(len(self.train_itr.dataset) / self.config.TRAIN.batch_size),
                    100 * (step) / int(len(self.train_itr.dataset) / self.config.TRAIN.batch_size),
                    ultimate_loss, *loss,
                    int(h), int(m), int(s)),
                # end="".center(20, ' '))
                end="")
        return np.array(total_loss).mean(0), np.array(total_acc).mean(0), np.sqrt(np.array(total_mse).mean(0))

    def eval(self, eval_itr, state=None):
        # loading models
        if state is not None:
            self.net.load_state_dict(state)
        else:
            try:
                state = torch.load(
                    self.config.ckpts_path + '/ckpt_{}_{}.pkl'.format(self.config.dataset, self.config.version)
                )
                self.net.load_state_dict(state['state_dict'])
            except:
                print("can't find the path to load state_dict from pretrained model!")
                exit()

        loss_fn = torch.nn.CrossEntropyLoss()
        metric_fn = multi_acc
        mse_fn = multi_mse
        total_loss = []
        total_acc = []
        total_mse = []
        for step, batch in enumerate(eval_itr):
            start_time = time.time()
            text, doc_length, sent_length = batch.text
            new_text = []
            # text (bs, sentence, word)
            for document in text:
                new_document = []
                for sentence in document:
                    new_document.append(
                        " ".join([self.train_itr.dataset.TEXT_FIELD.nesting_field.vocab.itos[word] for word in
                                  sentence]).replace("<pad>", "").strip())
                new_text.append(new_document)

            ts = [self.tokenizer.batch_encode_plus(i, padding='longest', max_length=512, truncation=True) for i in
                  new_text]
            unpacked_input_ids = [t.input_ids for t in ts]
            unpacked_attention_mask = [t.attention_mask for t in ts]
            length_sentence = [len(t.input_ids) for t in ts]
            length_word = [len(t.input_ids[0]) for t in ts]
            max_length_sentence = max(length_sentence)
            max_length_word = max(length_word)

            pad_sentence_template = [[[0] * max_length_word] * (max_length_sentence - l) for l in length_sentence]
            pad_word_template = [[0] * (max_length_word - l) for l in length_word]

            for i in range(len(unpacked_input_ids)):
                for j in range(len(unpacked_input_ids[i])):
                    unpacked_input_ids[i][j] += pad_word_template[i]
                    unpacked_attention_mask[i][j] += pad_word_template[i]

            for i in range(len(unpacked_input_ids)):
                unpacked_input_ids[i] += pad_sentence_template[i]
                unpacked_attention_mask[i] += pad_sentence_template[i]

            packed_input_ids = torch.Tensor(unpacked_input_ids).long().to(self.config.device)
            packed_attention_mask = torch.Tensor(unpacked_attention_mask).long().to(self.config.device)
            packed_embedding = []
            with torch.no_grad():
                for input_ids, attention_mask in zip(packed_input_ids, packed_attention_mask):
                    packed_embedding.append(self.bert(input_ids, attention_mask, output_hidden_states=True)[2][-2])
            packed_embedding = torch.stack(packed_embedding, 0)

            label = batch.label - 1
            usr = batch.usr
            prd = batch.prd

            # logits : logits_bup, logits_text, logits_prd, logits_usr, logits_all
            logits, atts = self.net(packed_embedding, usr, prd, mask=packed_attention_mask)
            if self.config.inter_sent_level:
                weights = self.config.TRAIN.alpha_bup, 0, 0, 0, self.config.TRAIN.alpha_all
            else:
                weights = self.config.TRAIN.alpha_bup, self.config.TRAIN.alpha_text, self.config.TRAIN.alpha_prd, self.config.TRAIN.alpha_usr, self.config.TRAIN.alpha_all

            loss = [loss_fn(logit, label) for logit in logits]
            metric = [metric_fn(label, logit) for logit in logits]
            metric_mse = [mse_fn(label, logit) for logit in logits]
            ultimate_loss = [l * w for l, w in zip(loss, weights)]
            ultimate_loss = reduce(lambda x, y: x + y, ultimate_loss)

            total_loss.append([ultimate_loss.data.cpu().numpy()] + [l.data.cpu().numpy() for l in loss])
            total_acc.append([m.data.cpu().numpy() for m in metric])
            total_mse.append([m.data.cpu().numpy() for m in metric_mse])

            # monitoring results on every steps
            end_time = time.time()
            span_time = (end_time - start_time) * (
                    int(len(eval_itr.dataset) / self.config.TRAIN.batch_size) - step)
            h = span_time // (60 * 60)
            m = (span_time % (60 * 60)) // 60
            s = (span_time % (60 * 60)) % 60 // 1
            print(
                "\rIteration: {:>4}/{} ({:>4.1f}%)   -ETA {:>2}h-{:>2}m-{:>2}s".format(
                    step, int(len(eval_itr.dataset) / self.config.TRAIN.batch_size),
                    100 * (step) / int(len(eval_itr.dataset) / self.config.TRAIN.batch_size),
                    int(h), int(m), int(s)),
                # end="".center(20, ' '))
                end="")

        return np.array(total_loss).mean(0), np.array(total_acc).mean(0), np.sqrt(np.array(total_mse).mean(0))

    def sample(self):
        text_field = self.train_itr.dataset.TEXT_FIELD
        # user_field = self.train_itr.dataset.USR_FIELD
        # item_field = self.train_itr.dataset.PRD_FIELD

        # data_sample = "this film is about a group of friends trying to help a forty year old man who is still a virgin . <sssss> `` the 40 year old virgin '' is a special film . <sssss> it has a lot of nude and sex scenes , many swear words and racist jokes , but amazingly it still comes across as adorable and fun . <sssss> andy is a well sculpted character , in the first scenes he is already shown to be a weirdo spending hours making a dish he does not eat during weekends . <sssss> yet he is strange in a cute way that makes viewers care about him . <sssss> his transformation is hilarious , and the ending is ever so sweet . <sssss> the film is so funny , one joke comes right after another that i can hardly stop laughing . <sssss> i have not been entertained by a movie so thoroughly for a long time ! "
        # user = "ur0035842/"
        # item = r"\tt0405422"
        # data_pre_process = text_field.preprocess(data_sample)
        # data_input =  text_field.process([data_pre_process])
        data_sample = "food is still good , maybe even better , the service is still the same , actually even worst . i still give the food 4 to 5 stars but service is negative 4 to 5 stars ."
        user = ["ivsDXr6e-z3sQBCMucK3XA", "LusAw6vTDC7KAfbuClMReA", "gzJpPaHN-NXBkAZcZri3hw", "sQwgFhirdCt6h_g1rbjHHA"]
        item = ["QiTjW1KT-pD5BKx9yYD85Q", "QiTjW1KT-pD5BKx9yYD85Q", "QiTjW1KT-pD5BKx9yYD85Q", "QiTjW1KT-pD5BKx9yYD85Q"]

        # data_pre_process = text_field.preprocess(data_sample)
        # data_pre_process_list = [data_pre_process]
        data_pre_process_list = [text_field.preprocess(data_sample) for _ in range(4)]
        data_input = text_field.process(data_pre_process_list)
        state = torch.load(
            self.config.ckpts_path + '/ckpt_{}_{}.pkl'.format(self.config.dataset, self.config.version)
        )
        self.net.load_state_dict(state['state_dict'])
        user_stoi = state['usr_vectors']['stoi']
        user_itos = state['usr_vectors']['itos']
        item_stoi = state['prd_vectors']['stoi']
        item_itos = state['prd_vectors']['itos']
        # print(user_stoi[user])
        # print(item_stoi[item])
        user_list = torch.Tensor([[user_stoi[i]] for i in user]).long().to(self.config.device)
        item_list = torch.Tensor([[item_stoi[i]] for i in item]).long().to(self.config.device)

        new_text = []
        for document in data_input[0]:
            new_document = []
            for sentence in document:
                new_document.append(
                    " ".join([self.train_itr.dataset.TEXT_FIELD.nesting_field.vocab.itos[word] for word in
                              sentence]).replace("<pad>", "").strip())
            new_text.append(new_document)

        ts = [self.tokenizer.batch_encode_plus(i, padding='longest', max_length=512, truncation=True) for i in
              new_text]
        unpacked_input_ids = [t.input_ids for t in ts]
        unpacked_attention_mask = [t.attention_mask for t in ts]
        length_sentence = [len(t.input_ids) for t in ts]
        length_word = [len(t.input_ids[0]) for t in ts]
        max_length_sentence = max(length_sentence)
        max_length_word = max(length_word)

        pad_sentence_template = [[[0] * max_length_word] * (max_length_sentence - l) for l in length_sentence]
        pad_word_template = [[0] * (max_length_word - l) for l in length_word]

        for i in range(len(unpacked_input_ids)):
            for j in range(len(unpacked_input_ids[i])):
                unpacked_input_ids[i][j] += pad_word_template[i]
                unpacked_attention_mask[i][j] += pad_word_template[i]

        for i in range(len(unpacked_input_ids)):
            unpacked_input_ids[i] += pad_sentence_template[i]
            unpacked_attention_mask[i] += pad_sentence_template[i]

        packed_input_ids = torch.Tensor(unpacked_input_ids).long().to(self.config.device)
        packed_attention_mask = torch.Tensor(unpacked_attention_mask).long().to(self.config.device)
        packed_embedding = []
        with torch.no_grad():
            for input_ids, attention_mask in zip(packed_input_ids, packed_attention_mask):
                packed_embedding.append(self.bert(input_ids, attention_mask, output_hidden_states=True)[2][-2])
        packed_embedding = torch.stack(packed_embedding, 0)

        logits, atts = self.net(packed_embedding,
                               user_list,
                               item_list,
                         mask=packed_attention_mask
                         )
        torch.save((packed_input_ids, atts, None), 'temp_bert.pkl')
        print(logits[4].shape)
        print(logits[4])
        print(torch.argmax(logits[4], -1))
        print(logits)
        print(atts)

    def get_attribute(self):
        text_field = self.train_itr.dataset.TEXT_FIELD
        state = torch.load(
            self.config.ckpts_path + '/ckpt_{}_{}.pkl'.format(self.config.dataset, self.config.version)
        )
        self.net.load_state_dict(state['state_dict'])
        user_stoi = state['usr_vectors']['stoi']
        user_itos = state['usr_vectors']['itos']
        item_stoi = state['prd_vectors']['stoi']
        item_itos = state['prd_vectors']['itos']

        users = ["iQTOwoJxHobk4kidfUIW9Q", "WWxCMDn8rVHIrIFoKRcRDg", "kH1fW7g_ZfURHsNkUb9TJA", "oy6fdscGSXY2gzRqF9pZxg", "Vi77s0AS-cIzPoBq5AbEaQ"]
        items = [
            ["ryvMJK6AlbU35HKrlFT61w", "4ntvolhaYeSCbHY1dhA53A", "BUnnBAtGkX05VJ6xI3y6JA", "8-4Qz-_g4kT-Eg1AekhXbA", "11sc9AN_zSKTjPxsage_iw", "bkm38p8gjvYqzNROj3oYrQ", "WWLSUH4M4DYw86mhvCKyeA", "Ht8mXLuqJSTPrU9kvzosUA", "0MN1z_ELvr4Tkl0OsLTVkg"],
            ["SLyacGZuMUKqrQJEooSkjw", "c0RSs2KYK5Y-ZlSrNq9LyA", "TgxDGx7L_JICWbuBUCGVqw", "CV4DDFG6tII-ehzaWPXK4g", "_oQN89kF_d97-cWW0iO8_g", "phe9voJ_LPAtqZaosM4Ibw"],
            ["DjOxXobyGDwWt89q4z1twg", "jRMyJkVWCtifZ7VCDUGdNA", "Yq8LiVymGA7vBpGCQuDfRw", "secsGLdQOaJPATB9SnHlew"],
            ["bzDs0u8I-z231QVdIQWkrA", "LzpR_jE6VIutJ08s2cdRrw", "9ziO3NpoNTKHvIKCBFB_fQ", "vwZ15OkVO6PemAe87k0M-Q", "MuIXnv7Oq7X3-4aEsp9dDA", "KgTb63IZHFn_rhLG-cpm_A", "9Y3aQAVITkEJYe5vLZr13w"],
            ["aYAlzKHwXQn6JNLweRnZjQ", "Nqcl3hDLyiwNQBxQpKCdIQ", "kC_cOPG3SvfWkcaQkHJW2w", "azXr0xlsOKDDiICIFnXi1Q", "mgN7A6Uw8ObmoTNZvx6VOg", "-K22CWorrReIV_kgLa6dmQ"]
        ]
        users_ids = [user_stoi[i] for i in users]
        items_ids = []
        for item in items:
            items_ids.append([item_stoi[i] for i in item])

        users_emebd = [self.net.usr_embedding(i) for i in users_ids]
        items_embed = []
        for item in items_ids:
            items.append([self.net.prd_embedding(i) for i in items_ids])
        print(users_emebd)
        print(items_embed)

    def run(self, run_mode):
        if run_mode == 'train':
            self.empty_log(self.config.version)
            self.train()
        elif run_mode == 'val':
            eval_loss, eval_acc, eval_rmse = self.eval(self.dev_itr, state=None)
            eval_logs = self.get_logging(eval_loss, eval_acc, eval_rmse, "evaluating")
            print("\r" + eval_logs)
        elif run_mode == 'test':
            eval_loss, eval_acc, eval_rmse = self.eval(self.test_itr, state=None)
            eval_logs = self.get_logging(eval_loss, eval_acc, eval_rmse, "evaluating")
            print("\r" + eval_logs)
        elif run_mode == 'sample':
            self.sample()
        elif run_mode == 'attr':
            self.get_attribute()
        else:
            exit(-1)

    def empty_log(self, version):
        if (os.path.exists(self.config.log_path + '/log_run_' + self.config.dataset + '_' + version + '.txt')):
            os.remove(self.config.log_path + '/log_run_' + self.config.dataset + '_' + version + '.txt')
        print('Initializing log file ........')
        print('Finished!')
        print('')

    def logging(self, log_file, logs):
        logfile = open(
            log_file, 'a+'
        )
        logfile.write(logs)
        logfile.close()

    def get_logging(self, loss, acc, rmse, eval='training'):
        logs = \
            '==={} phrase...'.format(eval) + "".center(60, " ") + "\n" + \
            '\t'.join(["{:<10}"] * 6).format("total_loss", "loss_bup", "loss_text", "loss_prd", "loss_usr",
                                             "loss_all") + '\n' + \
            '\t'.join(["{:^10.3f}"] * 6).format(loss[-1], *loss) + '\n' + \
            '\t'.join(["{:<10}"] * 6).format("total_acc", "acc_bup", "acc_text", "acc_prd", "acc_usr",
                                             "acc_all") + '\n' + \
            '\t'.join(["{:^10.3f}"] * 6).format(acc[acc.argmax()], *acc) + '\n' + \
            '\t'.join(["{:<10}"] * 6).format("total_rmse", "rmse_bup", "rmse_text", "rmse_prd", "rmse_usr",
                                             "rmse_all") + '\n' + \
            '\t'.join(["{:^10.3f}"] * 6).format(rmse[rmse.argmin()], *rmse) + '\n'

        return logs
