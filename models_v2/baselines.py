import torch
import torch.nn as nn
import torch.nn.functional as F
from models_v2.baseline_layers.rnn import WordLevelRNN, WordLevelRNN_LA, SentLevelRNN, SentLevelRNN_LA, SentLevelRNN_UoP, WordLevelRNN_UoP
from models_v2.classifier import Classifier

class LSTM(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.text_embedding = config.text_embedding
        self.text_embed = nn.Embedding(config.text_embedding.size(0), config.text_embedding.size(1),
                                       padding_idx=config.pad_idx)
        self.text_embed.weight.data.copy_(self.text_embedding)
        self.text_embed.weight.requires_grad = False
        if config.bidirectional == True:
            self.lstm = nn.LSTM(config.embed_dim, config.word_hidden_dim // 2, batch_first=True, bidirectional=True)
        else:
            self.lstm = nn.LSTM(config.embed_dim, config.word_hidden_dim, batch_first=True)
        self.classifier = Classifier(config.word_hidden_dim, config)

    def forward(self, input_ids, input_lengths=None):
        input_embeds = self.text_embed(input_ids)
        if not hasattr(self, '_flattened'):
            self.lstm.flatten_parameters()
            setattr(self, '_flattened', True)
        mask = self.generate_mask(input_lengths, input_ids.shape[1]).unsqueeze(2) # (bs, seq, 1)
        hidden_states, _ = self.lstm(input_embeds) # (bs, seq, dim), _
        hidden_state = (hidden_states * mask).sum(1)
        logits = self.classifier(hidden_state)
        return logits

    def generate_mask(self, input_lengths, num_classes):
        # input_lengths (bs,)
        mask = torch.nn.functional.one_hot(input_lengths-1, num_classes=num_classes) # (bs, seq)
        return mask


class NSC_LA(torch.nn.Module):
    def __init__(self, config):
        super(NSC_LA, self).__init__()
        self.config = config
        self.text_embedding = config.text_embedding
        self.text_embed = nn.Embedding(config.text_embedding.size(0), config.text_embedding.size(1),
                                       padding_idx=config.pad_idx)
        self.text_embed.weight.data.copy_(self.text_embedding)
        self.text_embed.weight.requires_grad = False
        self.word_attention_rnn = WordLevelRNN_LA(config)
        self.sentence_attention_rnn = SentLevelRNN_LA(config)
        self.classifier_sents = Classifier(config.sent_hidden_dim, config)

    def forward(self, input_ids, user, prd, mask=None):
        input_ids = input_ids.permute(1, 0, 2)  # input_ids: (sent, batch, word)
        input_embeds = self.text_embed(input_ids) # input_embeds: (sent, batch, word_dim)
        num_sentences = input_embeds.size(0)
        words_text = []

        mask_word = mask.permute(1, 0, 2)  # text: (sent, batch, word)
        mask_sent = mask.long().sum(2) > 0  # (batch, sent)

        for i in range(num_sentences):
            text_word = self.word_attention_rnn(input_embeds[i], mask=mask_word[i])
            words_text.append(text_word)
        words_text = torch.stack(words_text, 1) # (batch, sents, dim)

        sents = self.sentence_attention_rnn(words_text, mask=mask_sent)
        logits = self.classifier_sents(sents)
        return logits


class NSC_UPA(torch.nn.Module):
    def __init__(self, config):
        super(NSC_UPA, self).__init__()
        self.config = config
        self.text_embedding = config.text_embedding
        self.text_embed = nn.Embedding(config.text_embedding.size(0), config.text_embedding.size(1),
                                       padding_idx=config.pad_idx)
        self.text_embed.weight.data.copy_(self.text_embedding)
        self.text_embed.weight.requires_grad = False

        self.usr_embed = nn.Embedding(config.usr_embedding.size(0), config.usr_dim,
                                      padding_idx=config.pad_idx)
        # self.usr_embed.weight.data.copy_(self.usr_embedding)
        self.usr_embed.weight.data.copy_(
            torch.Tensor(config.usr_embedding.size(0), config.usr_dim).uniform_(-0.01, 0.01))
        # self.usr_embed.weight.data.copy_(
        #     torch.Tensor(config.usr_embedding.size(0), config.usr_dim).zero_())
        self.usr_embed.weight.requires_grad = True

        self.prd_embed = nn.Embedding(config.prd_embedding.size(0), config.prd_dim,
                                      padding_idx=config.pad_idx)
        # self.prd_embed.weight.data.copy_(self.prd_embedding)
        self.prd_embed.weight.data.copy_(
            torch.Tensor(config.prd_embedding.size(0), config.prd_dim).uniform_(-0.01, 0.01))
        # self.prd_embed.weight.data.copy_(
        #     torch.Tensor(config.prd_embedding.size(0), config.prd_dim).zero_())
        self.prd_embed.weight.requires_grad = True

        self.word_attention_rnn = WordLevelRNN(config)
        self.sentence_attention_rnn = SentLevelRNN(config)
        self.classifier_sents = Classifier(config.sent_hidden_dim, config)

    def forward(self, input_ids, usr, prd, mask=None):
        input_ids = input_ids.permute(1, 0, 2)  # input_ids: (sent, batch, word)
        input_embeds = self.text_embed(input_ids) # input_embeds: (sent, batch, word_dim)
        usr_embeds = self.usr_embed(usr)
        prd_embeds = self.prd_embed(prd)
        num_sentences = input_embeds.size(0)
        words_text = []

        mask_word = mask.permute(1, 0, 2)  # text: (sent, batch, word)
        mask_sent = mask.long().sum(2) > 0  # (batch, sent)

        for i in range(num_sentences):
            text_word = self.word_attention_rnn(input_embeds[i], usr_embeds, prd_embeds, mask=mask_word[i])
            words_text.append(text_word)
        words_text = torch.stack(words_text, 1) # (batch, sents, dim)

        sents = self.sentence_attention_rnn(words_text, usr_embeds, prd_embeds, mask=mask_sent)
        logits = self.classifier_sents(sents)
        return logits


class KimCNN(nn.Module):

    def __init__(self, config):
        super().__init__()
        output_channel = 200
        words_dim = 300
        # ks = 3 # There are three conv nets here

        input_channel = 1
        self.text_embedding = config.text_embedding
        self.text_embedding = nn.Embedding.from_pretrained(self.text_embedding, freeze=False)

        self.conv1 = nn.Conv2d(input_channel, output_channel, (1, words_dim))
        self.conv2 = nn.Conv2d(input_channel, output_channel, (2, words_dim), padding=(1,0))
        self.conv3 = nn.Conv2d(input_channel, output_channel, (3, words_dim), padding=(2,0))

        self.dropout = nn.Dropout(config.dropout)
        self.fc1 = nn.Linear(output_channel, config.num_classes)

    def forward(self, x, usr, prd, **kwargs):

        non_static_input = self.text_embedding(x)
        x = non_static_input.unsqueeze(1) # (batch, channel_input, sent_len, embed_dim)

        x = [F.relu(self.conv1(x)).squeeze(3), F.relu(self.conv2(x)).squeeze(3), F.relu(self.conv3(x)).squeeze(3)]
        # (batch, channel_output, ~=sent_len) * ks
        x = [F.max_pool1d(i, i.size(2)).squeeze(2) for i in x] # max-over-time pooling
        # (batch, channel_output) * ks
        # x = torch.cat(x, 1) # (batch, channel_output * ks)
        x = torch.stack(x, 0).sum(0) # (batch, channel_output)
        x = self.dropout(x)
        logit = self.fc1(x) # (batch, target_size)
        return logit


class UPNN_CNN(nn.Module):

    def __init__(self, config):
        super().__init__()
        output_channel = config.output_channel
        words_dim = config.words_dim
        # ks = 3 # There are three conv nets here

        self.config = config
        self.text_embedding = config.text_embedding
        self.text_embed = nn.Embedding(config.text_embedding.size(0), config.text_embedding.size(1),
                                       padding_idx=config.pad_idx)
        self.text_embed.weight.data.copy_(self.text_embedding)
        # self.text_embed.weight.requires_grad = False
        # self.text_embed.weight.data.copy_(torch.Tensor(config.text_embedding.size(0), config.text_embedding.size(1)).uniform_(-0.01, 0.01))
        self.text_embed.weight.requires_grad = True

        self.usr_embed = nn.Embedding(config.usr_embedding.size(0), config.usr_dim,
                                      padding_idx=config.pad_idx)
        self.usr_embed.weight.data.copy_(
            torch.Tensor(config.usr_embedding.size(0), config.usr_dim).uniform_(-0.01, 0.01))
        self.usr_embed.weight.requires_grad = True

        self.usr_matrix_embed = nn.Embedding(config.usr_embedding.size(0), config.words_hidden_size*config.words_dim,
                                      padding_idx=config.pad_idx)
        self.usr_matrix_embed.weight.data.copy_(
            torch.Tensor(config.usr_embedding.size(0), config.words_hidden_size*config.words_dim).uniform_(-0.01, 0.01))
        self.usr_matrix_embed.weight.requires_grad = True

        self.prd_embed = nn.Embedding(config.prd_embedding.size(0), config.prd_dim,
                                      padding_idx=config.pad_idx)
        self.prd_embed.weight.data.copy_(
            torch.Tensor(config.prd_embedding.size(0), config.prd_dim).uniform_(-0.01, 0.01))
        self.prd_embed.weight.requires_grad = True

        self.prd_matrix_embed = nn.Embedding(config.prd_embedding.size(0), config.words_hidden_size * config.words_dim,
                                             padding_idx=config.pad_idx)
        self.prd_matrix_embed.weight.data.copy_(
            torch.Tensor(config.prd_embedding.size(0), config.words_hidden_size * config.words_dim).uniform_(-0.01, 0.01))
        self.usr_matrix_embed.weight.requires_grad = True

        input_channel = 1
        # self.text_embedding = config.text_embedding
        # self.text_embedding = nn.Embedding.from_pretrained(self.text_embedding, freeze=False)

        self.linear = nn.Linear(config.words_hidden_size * 2, config.words_hidden_size * 2)

        self.conv1 = nn.Conv2d(input_channel, output_channel, (1, config.words_dim))
        self.conv2 = nn.Conv2d(input_channel, output_channel, (2, config.words_dim), padding=(1,0))
        self.conv3 = nn.Conv2d(input_channel, output_channel, (3, config.words_dim), padding=(2,0))

        self.dropout = nn.Dropout(config.dropout)
        self.fc1 = nn.Linear(output_channel+self.config.usr_dim+self.config.prd_dim, config.num_classes)
        # self.fc1 = nn.Linear(output_channel, config.num_classes)

    def forward(self, x, usr, prd, **kwargs):
        if len(usr.size()) > 1:
            usr = usr.squeeze()
            prd = prd.squeeze() 
        x = self.text_embed(x) # (batch, sent_len, embed_dim)
        batch_size = x.size(0)

        usr_embeds = self.usr_embed(usr)
        prd_embeds = self.prd_embed(prd)
        if batch_size == 1:
            usr_embeds = usr_embeds.unsqueeze(0)
            prd_embeds = prd_embeds.unsqueeze(0)

        # usr_matrix = self.usr_matrix_embed(usr).reshape(batch_size, self.config.words_hidden_size, self.config.words_dim) # (bs, words_hidden_size, word_dim)
        # prd_matrix = self.prd_matrix_embed(prd).reshape(batch_size, self.config.words_hidden_size, self.config.words_dim)
        #
        # # # (bs, word_dim, word_dim) * (bs, seq, word_dim) -> (bs, seq, word_dim)
        # x = torch.cat([torch.einsum('abc,adc->adb', usr_matrix, x), torch.einsum('abc,adc->adb', prd_matrix, x)], -1) # (bs, seq, dim*2)
        # x = self.linear(x)
        # x = torch.tanh(x)
        x = x.unsqueeze(1)  # (batch, channel_input, sent_len, embed_dim*2)
        x = [F.relu(self.conv1(x)).squeeze(3), F.relu(self.conv2(x)).squeeze(3), F.relu(self.conv3(x)).squeeze(3)]
        # (batch, channel_output, ~=sent_len) * ks

        x = [F.max_pool1d(i, i.size(2)).squeeze(2) for i in x] # max-over-time pooling
        # (batch, channel_output) * ks
        # x = torch.cat(x, 1) # (batch, channel_output * ks)
        x = torch.stack(x, 0).sum(0) # (batch, channel_output)
        x = torch.cat([x, usr_embeds, prd_embeds], 1) # (batch, chanel_output+usr_dim+prd_dim)
        x = self.dropout(x)
        logit = self.fc1(x) # (batch, target_size)
        return logit


class UPNN_NSC(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.text_embedding = config.text_embedding
        self.text_embed = nn.Embedding(config.text_embedding.size(0), config.text_embedding.size(1),
                                       padding_idx=config.pad_idx)
        self.text_embed.weight.data.copy_(self.text_embedding)
        self.text_embed.weight.requires_grad = True

        self.usr_embed = nn.Embedding(config.usr_embedding.size(0), config.usr_dim,
                                      padding_idx=config.pad_idx)
        self.usr_embed.weight.data.copy_(
            torch.Tensor(config.usr_embedding.size(0), config.usr_dim).uniform_(-0.01, 0.01))
        self.usr_embed.weight.requires_grad = True

        self.usr_matrix_embed = nn.Embedding(config.usr_embedding.size(0), config.word_hidden_dim*config.words_dim // 2,
                                      padding_idx=config.pad_idx)
        self.usr_matrix_embed.weight.data.copy_(
            torch.Tensor(config.usr_embedding.size(0), config.word_hidden_dim*config.words_dim // 2).uniform_(-0.01, 0.01))
        self.usr_matrix_embed.weight.requires_grad = True

        self.prd_embed = nn.Embedding(config.prd_embedding.size(0), config.prd_dim,
                                      padding_idx=config.pad_idx)
        self.prd_embed.weight.data.copy_(
            torch.Tensor(config.prd_embedding.size(0), config.prd_dim).uniform_(-0.01, 0.01))
        self.prd_embed.weight.requires_grad = True

        self.prd_matrix_embed = nn.Embedding(config.prd_embedding.size(0), config.word_hidden_dim * config.words_dim // 2,
                                             padding_idx=config.pad_idx)
        self.prd_matrix_embed.weight.data.copy_(
            torch.Tensor(config.prd_embedding.size(0), config.word_hidden_dim * config.words_dim // 2).uniform_(-0.01, 0.01))
        self.usr_matrix_embed.weight.requires_grad = True

        self.linear = nn.Linear(config.word_hidden_dim, config.words_dim)
        self.word_attention_rnn = WordLevelRNN(config)
        self.sentence_attention_rnn = SentLevelRNN(config)
        self.fc1 = nn.Linear(config.sent_hidden_dim + self.config.usr_dim + self.config.prd_dim, config.num_classes)

    def forward(self, input_ids, usr, prd, mask=None):
        batch_size = input_ids.size(0)
        input_ids = input_ids.permute(1, 0, 2)  # input_ids: (sent, batch, word)
        input_embeds = self.text_embed(input_ids)  # input_embeds: (sent, batch, word_dim)
        usr_embeds = self.usr_embed(usr) # (batch, dim)
        prd_embeds = self.prd_embed(prd)
        if len(usr.size()) > 1:
            usr = usr.squeeze()
            prd = prd.squeeze()

        usr_matrix = self.usr_matrix_embed(usr).reshape(batch_size, self.config.word_hidden_dim // 2, self.config.words_dim) # (bs, words_hidden_size, word_dim)
        prd_matrix = self.prd_matrix_embed(prd).reshape(batch_size, self.config.word_hidden_dim // 2, self.config.words_dim)

        # # (bs, word_dim, word_dim) * (sent, bs, seq, word_dim) -> (sent, bs, seq, word_dim)
        input_embeds = torch.cat([torch.einsum('abc,eadc->eadb', usr_matrix, input_embeds), torch.einsum('abc,eadc->eadb', prd_matrix, input_embeds)], -1) # (bs, seq, dim*2)
        input_embeds = self.linear(input_embeds)
        input_embeds = torch.tanh(input_embeds)

        num_sentences = input_embeds.size(0)
        words_text = []

        mask_word = mask.permute(1, 0, 2)  # text: (sent, batch, word)
        mask_sent = mask.long().sum(2) > 0  # (batch, sent)

        for i in range(num_sentences):
            text_word = self.word_attention_rnn(input_embeds[i], usr_embeds, prd_embeds, mask=mask_word[i])
            words_text.append(text_word)
        words_text = torch.stack(words_text, 1)  # (batch, sents, dim)
        sents = self.sentence_attention_rnn(words_text, usr_embeds, prd_embeds, mask=mask_sent)

        try:
            x = torch.cat([sents, usr_embeds.squeeze(1), prd_embeds.squeeze(1)], 1)  # (batch, chanel_output+usr_dim+prd_dim)
        except:
            print(sents.shape)
            print(usr_embeds.shape)
            print(prd_embeds.shape)
        logits = self.fc1(x)
        return logits


class HUAPA(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.text_embedding = config.text_embedding
        self.usr_embedding = config.usr_embedding
        self.prd_embedding = config.prd_embedding
        self.text_embed = nn.Embedding(config.text_embedding.size(0), config.text_embedding.size(1),
                                       padding_idx=config.pad_idx)
        self.text_embed.weight.data.copy_(self.text_embedding)
        self.text_embed.weight.requires_grad = False

        self.usr_embed = nn.Embedding(config.usr_embedding.size(0), config.usr_dim,
                                      padding_idx=config.pad_idx)
        self.usr_embed.weight.data.copy_(torch.Tensor(config.usr_embedding.size(0), config.usr_dim).uniform_(-0.01,0.01))
        self.usr_embed.weight.requires_grad = True

        self.prd_embed = nn.Embedding(config.prd_embedding.size(0), config.prd_dim,
                                      padding_idx=config.pad_idx)
        self.prd_embed.weight.data.copy_(torch.Tensor(config.prd_embedding.size(0), config.prd_dim).uniform_(-0.01,0.01))
        self.prd_embed.weight.requires_grad = True

        self.word_attention_rnn = WordLevelRNN_UoP(config)
        self.sentence_attention_rnn = SentLevelRNN_UoP(config)
        self.classifier_all = Classifier(config.sent_hidden_dim * 2, config)
        self.classifier_usr = Classifier(config.sent_hidden_dim, config)
        self.classifier_prd = Classifier(config.sent_hidden_dim, config)

    def forward(self, text, usr, prd, mask=None):
        # text: (batch, sent, word)
        # usr: (batch, 1)
        # prd: (batch, 1)
        # mask: (batch, sent, word)
        text = text.permute(1, 0, 2)  # text: (sent, batch, word)
        usr = usr.squeeze(-1)  # usr: (batch, )
        prd = prd.squeeze(-1)  # prd: (batch, )

        # word embedding
        text = self.text_embed(text)
        usr = self.usr_embed(usr)
        prd = self.prd_embed(prd)

        mask_word = None
        mask_sent = None
        if mask is not None:
            mask_word = mask.permute(1, 0, 2)  # text: (sent, batch, word)
            mask_sent = mask.long().sum(2) > 0  # (batch, sent)

        num_sentences = text.size(0)
        words_usr = []
        words_prd = []
        for i in range(num_sentences):
            usr_attn, prd_attn = self.word_attention_rnn(text[i], usr, prd, mask=mask_word[i])
            words_usr.append(usr_attn)
            words_prd.append(prd_attn)
        words_usr = torch.stack(words_usr, 1)  # (batch, sents, dim)
        words_prd = torch.stack(words_prd, 1)  # (batch, sents, dim)

        sents_usr, sents_prd = self.sentence_attention_rnn(words_usr, words_prd, usr, prd,
                                                                   mask=mask_sent)
        logits_usr = self.classifier_usr(sents_usr)
        logits_prd = self.classifier_prd(sents_prd)
        logits_all = self.classifier_all(torch.cat([sents_usr, sents_prd], dim=-1))
        return logits_usr, logits_prd, logits_all