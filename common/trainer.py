import os
import time
import torch
import datetime
import numpy as np
from functools import reduce
from models_v2.get_optim import get_Adam_optim, get_optim
from models_v2.model import Net
from common.utils import load_vectors, multi_acc, multi_mse


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
            label = batch.label - 1
            usr = batch.usr
            prd = batch.prd

            # logits : logits_bup, logits_text, logits_prd, logits_usr, logits_all
            logits, atts = self.net(text, usr, prd, mask=(text != self.config.pad_idx))
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
            label = batch.label - 1
            usr = batch.usr
            prd = batch.prd

            # logits : logits_bup, logits_text, logits_prd, logits_usr, logits_all
            logits, atts = self.net(text, usr, prd, mask=(text != self.config.pad_idx))
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

            # monitoring results on every steps1
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
        # data_sample = "this film is about a group of friends trying to help a forty year old man who is still a virgin . <sssss> `` the 40 year old virgin '' is a special film . <sssss> it has a lot of nude and sex scenes , many swear words and racist jokes , but amazingly it still comes across as adorable and fun . <sssss> andy is a well sculpted character , in the first scenes he is already shown to be a weirdo spending hours making a dish he does not eat during weekends . <sssss> yet he is strange in a cute way that makes viewers care about him . <sssss> his transformation is hilarious , and the ending is ever so sweet . <sssss> the film is so funny , one joke comes right after another that i can hardly stop laughing . <sssss> i have not been entertained by a movie so thoroughly for a long time ! "

        # data_sample = "this place certainly deserves all the praise they can get . <sssss> all the other non-5 star reviews are just haters who are claiming they were underwhelmed . <sssss> i must admit this is the best gelato i 've ever had . <sssss> although i 've never been to italy i think i 've had my fair share of gelato . <sssss> you will see she has an array of different flavors and wo n't mind giving you a sample of all them , if you really needed it ? <sssss> she is that patient that she does n't mind doing this . <sssss> some of the more interesting flavors i 've seen are rosemary and maple bacon . <sssss> however , on both visits i 've had the one with white chocolate kit kat because it 's so good ! <sssss> i love white chocolate kit kats and this gelato version is absolutely divine . <sssss> the texture is so smooth and creamy . <sssss> the inside of this place is really clean and all the fixtures look new . <sssss> although this is an old shopping center she did a fine job of making it up to date . <sssss> if she had a predominant sign on the street it would definitely bring her more business . <sssss> art of flavors is definitely a culinary destination ."
        # item = r"Nu_IcBFRt63p2OHzF2hUig"

        # data_sample = "i did n't know cake boss had a restaurant here and i actually found it through instagram . <sssss> my friend and i came on saturday and we sat at the bar since it was happy hour . <sssss> we ordered the sausage and pepper sliders , mac n cheese and italian birthday cake . <sssss> and we were very pleased with our order . <sssss> the sliders tasted like what you would expect typical sausage and peppers , but tasty . <sssss> mac n cheese had chicken and veggies , which is more like a entree versus what most americans think of it being a side dish . <sssss> service was standard and considering we sat at the bar i 've never had a bartender treat me poorly , but i must admit she was kind of awkward towards us . <sssss> lastly , the italian birthday cake was really moist , but not dippy . <sssss> i ca n't wait to come here again and try the chicken parmigiana , a customer next to us ordered this and it looked delicious . <sssss> i highly recommend buddy v 's to anyone especially if you 're at venetian already ."
        # user = ["iQTOwoJxHobk4kidfUIW9Q", "LuXmwayVAlcwreN3sH7MzQ", "YQeIDbh6UR9vWAiJomoEKQ", "sovduOweKgPFAwA-MmmtzA", "VZWOi9UAqveUbQxbapKB7g", "thdVzCfKx-DV0zYWqId3pw"]
        # item = [r"y8VQQO_WkYNjSLcq6hyjPA", r"y8VQQO_WkYNjSLcq6hyjPA", r"y8VQQO_WkYNjSLcq6hyjPA",  r"y8VQQO_WkYNjSLcq6hyjPA", r"y8VQQO_WkYNjSLcq6hyjPA", r"y8VQQO_WkYNjSLcq6hyjPA"]

        data_sample = "great cheap quality food, this place gets packed fast !!"
        user = ["9VmTOyq01oIUk5zuxOj1GA", "LusAw6vTDC7KAfbuClMReA", "gzJpPaHN-NXBkAZcZri3hw", "sQwgFhirdCt6h_g1rbjHHA"]
        item = ["g9jX2oXQr8zrOQgynYnYjQ", "g9jX2oXQr8zrOQgynYnYjQ", "g9jX2oXQr8zrOQgynYnYjQ", "g9jX2oXQr8zrOQgynYnYjQ"]

        # data_pre_process = text_field.preprocess(data_sample)
        # data_pre_process_list = [data_pre_process]
        data_pre_process_list = [text_field.preprocess(data_sample) for _ in range(4)]
        data_input =  text_field.process(data_pre_process_list)
        input_ids = data_input[0].to(self.config.device)
        state = torch.load(
            self.config.ckpts_path + '/ckpt_{}_{}.pkl'.format(self.config.dataset, self.config.version)
        )
        self.net.load_state_dict(state['state_dict'])
        user_stoi = state['usr_vectors']['stoi']
        user_itos = state['usr_vectors']['itos']
        item_stoi = state['prd_vectors']['stoi']
        item_itos = state['prd_vectors']['itos']

        user_list = torch.Tensor([[user_stoi[i]] for i in user]).long().to(self.config.device)
        item_list = torch.Tensor([[item_stoi[i]] for i in item]).long().to(self.config.device)
        mask = (input_ids != self.config.pad_idx)
        print(input_ids.shape)
        print(user_list.shape)
        print(item_list.shape)
        print(mask.shape)
        logits, atts = self.net(input_ids,
                         user_list,
                         item_list,
                         mask=mask
                         )
        print("*"*30)
        for i in atts:
            print(i.shape)
        print("*"*30)
        atts = [a.cpu().detach() for a in atts]
        torch.save((input_ids, atts, self.train_itr.dataset.TEXT_FIELD.nesting_field.vocab.itos), 'temp.pkl')
        print("-"*20)
        print(logits[4].shape)
        print(logits[4])
        print(torch.argmax(logits[4], -1))
        print(atts)
        print(atts[0].shape)

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

        users = ["iQTOwoJxHobk4kidfUIW9Q", "WWxCMDn8rVHIrIFoKRcRDg", "kH1fW7g_ZfURHsNkUb9TJA", "oy6fdscGSXY2gzRqF9pZxg", "Vi77s0AS-cIzPoBq5AbEaQ", "kSVYpNWA19wUplbdi0U0Uw", "J5nb9AT1LDiGN4_hY0Au8Q", "q9XgOylNsSbqZqF_SO3-OQ"]
        items = [
            ["ryvMJK6AlbU35HKrlFT61w", "4ntvolhaYeSCbHY1dhA53A", "BUnnBAtGkX05VJ6xI3y6JA", "8-4Qz-_g4kT-Eg1AekhXbA", "11sc9AN_zSKTjPxsage_iw", "bkm38p8gjvYqzNROj3oYrQ", "WWLSUH4M4DYw86mhvCKyeA", "Ht8mXLuqJSTPrU9kvzosUA", "0MN1z_ELvr4Tkl0OsLTVkg"],
            ["SLyacGZuMUKqrQJEooSkjw", "c0RSs2KYK5Y-ZlSrNq9LyA", "TgxDGx7L_JICWbuBUCGVqw", "CV4DDFG6tII-ehzaWPXK4g", "_oQN89kF_d97-cWW0iO8_g", "phe9voJ_LPAtqZaosM4Ibw"],
            ["DjOxXobyGDwWt89q4z1twg", "jRMyJkVWCtifZ7VCDUGdNA", "Yq8LiVymGA7vBpGCQuDfRw", "secsGLdQOaJPATB9SnHlew"],
            ["bzDs0u8I-z231QVdIQWkrA", "LzpR_jE6VIutJ08s2cdRrw", "9ziO3NpoNTKHvIKCBFB_fQ", "vwZ15OkVO6PemAe87k0M-Q", "MuIXnv7Oq7X3-4aEsp9dDA", "KgTb63IZHFn_rhLG-cpm_A", "9Y3aQAVITkEJYe5vLZr13w"],
            ["aYAlzKHwXQn6JNLweRnZjQ", "Nqcl3hDLyiwNQBxQpKCdIQ", "kC_cOPG3SvfWkcaQkHJW2w", "azXr0xlsOKDDiICIFnXi1Q", "mgN7A6Uw8ObmoTNZvx6VOg", "-K22CWorrReIV_kgLa6dmQ"],
            ["A_vRqMo7HrlnCRFvzWVcfw", "OVU0pQ85Zf74OVgDyPFvGQ", "lfmvmL6iHq9-IDABLeNsuw", "n6oNY8L8iEWkRksJkmwCqg", "dNAyji6q-uZcnFByNzZ_2g"],
            ["MuIXnv7Oq7X3-4aEsp9dDA", "LVngid2NNh2s5cAjuOw6tw", "57YV3wsiNBp-aK25qjUeww", "8HQ8clouLGgee99KkR4vXA"],
            ["7tTK3VPlFtBGBHm8-LZIUg", "1MQmujTuU-3qPdoogdA8CQ", "NDKkce5Au-o_OhIt5f2ZBg", "uMxfxbqT27XDeU9UTXeErg", "Ioi8SAipW_eLRuwH4BV0_Q", "Q0AADLgsYi1sFDk8jtMYUw", "_FXql6eVhbM923RdCi94SA", "EWMwV5V9BxNs_U6nNVMeqw", "WS1z1OAR0tRl4FsjdTGUFQ", "DDnmNTvIIQu2t3WZ2EQx-w"]
        ]
        users_ids = [user_stoi[i] for i in users]
        items_ids = []
        for item in items:
            items_ids.append([item_stoi[i] for i in item])

        # print(self.net.module.usr_embed(torch.Tensor([users_ids[0]]).long()))
        self.net.to('cpu')
        users_emebd = self.net.module.usr_embed(torch.LongTensor(users_ids))
        items_embed = []
        for item in items_ids:
            items_embed.append(self.net.module.prd_embed(torch.LongTensor(item)))

        torch.save((users_emebd, items_embed, self.net.module.prd_embed.weight), 'attr.pkl')


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
