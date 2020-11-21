import torch
import torch.nn as nn
from models_v2.baseline_layers.baseline_att import ConcateAttention, SelfAttention, UoPAttention, UoPConcatAttention, BilinearAttention

class WordLevelRNN(nn.Module):
    def __init__(self, config):
        super().__init__()
        words_dim = config.words_dim
        word_hidden_dim = config.word_hidden_dim
        usr_dim = config.usr_dim
        prd_dim = config.prd_dim

        self.gru = nn.LSTM(words_dim, word_hidden_dim // 2, bidirectional=True, batch_first=True)

        self.concate_att = ConcateAttention(config, word_hidden_dim, usr_dim, prd_dim)
        # self.usr_att = BilinearAttention(usr_dim, word_hidden_dim, config)

    def forward(self, input_embeded, usr, prd, mask=None):
        if not hasattr(self, '_flattened'):
            self.gru.flatten_parameters()
            setattr(self, '_flattened', True)
        hidden_states, _ = self.gru(input_embeded)
        output = self.concate_att(hidden_states, usr, prd, mask=mask)
        # output = self.usr_att(usr, hidden_states, mask=mask)
        return output

class SentLevelRNN(nn.Module):

    def __init__(self, config):
        super().__init__()
        sentence_hidden_dim = config.sent_hidden_dim
        word_hidden_dim = config.word_hidden_dim
        usr_dim = config.usr_dim
        prd_dim = config.prd_dim

        self.gru = nn.LSTM(word_hidden_dim, sentence_hidden_dim // 2, bidirectional=True, batch_first=True)

        self.concate_att = ConcateAttention(config, sentence_hidden_dim, usr_dim, prd_dim)
        # self.usr_att = BilinearAttention(usr_dim, sentence_hidden_dim, config)

    def forward(self, input_embeded, usr, prd, mask=None):
        if not hasattr(self, '_flattened'):
            self.gru.flatten_parameters()
            setattr(self, '_flattened', True)
        hidden_states, _ = self.gru(input_embeded)
        output = self.concate_att(hidden_states, usr, prd, mask=mask)
        # output = self.usr_att(usr, hidden_states, mask=mask)
        return output

class WordLevelRNN_LA(nn.Module):

    def __init__(self, config):
        super().__init__()
        words_dim = config.words_dim
        word_hidden_dim = config.word_hidden_dim

        self.gru = nn.LSTM(words_dim, word_hidden_dim // 2, bidirectional=True, batch_first=True)

        self.att = SelfAttention(word_hidden_dim, config)

    def forward(self, input_embeded, mask=None):
        if not hasattr(self, '_flattened'):
            self.gru.flatten_parameters()
            setattr(self, '_flattened', True)
        hidden_states, _ = self.gru(input_embeded)
        output = self.att(hidden_states, mask=mask)
        return output

class SentLevelRNN_LA(nn.Module):

    def __init__(self, config):
        super().__init__()
        sentence_hidden_dim = config.sent_hidden_dim
        word_hidden_dim = config.word_hidden_dim

        self.gru = nn.LSTM(word_hidden_dim, sentence_hidden_dim // 2, bidirectional=True, batch_first=True)

        self.att = SelfAttention(sentence_hidden_dim, config)


    def forward(self, input_embeded, mask=None):
        if not hasattr(self, '_flattened'):
            self.gru.flatten_parameters()
            setattr(self, '_flattened', True)
        hidden_states, _ = self.gru(input_embeded)
        output = self.att(hidden_states, mask=mask)
        return output

class SentLevelRNN_UoP(nn.Module):
    def __init__(self, config):
        super().__init__()
        sentence_hidden_dim = config.sent_hidden_dim
        word_hidden_dim = config.word_hidden_dim
        usr_dim = config.usr_dim
        prd_dim = config.prd_dim
        self.usr_gru = nn.LSTM(word_hidden_dim, sentence_hidden_dim // 2, bidirectional=True, batch_first=True)
        self.prd_gru = nn.LSTM(word_hidden_dim, sentence_hidden_dim // 2, bidirectional=True, batch_first=True)

        self.usr_att = UoPAttention(usr_dim, sentence_hidden_dim, config)
        # self.usr_att = BilinearAttention(usr_dim, sentence_hidden_dim, config)
        # self.prd_att = UoPAttention(prd_dim, sentence_hidden_dim, config)
        self.prd_att = BilinearAttention(prd_dim, sentence_hidden_dim, config)
        self.layernorm = nn.LayerNorm(sentence_hidden_dim)

    def forward(self, usr_x, prd_x, usr, prd, mask=None):
        if not hasattr(self, '_flattened'):
            self.usr_gru.flatten_parameters()
            self.prd_gru.flatten_parameters()
            setattr(self, '_flattened', True)
        h_usr, _ = self.usr_gru(usr_x)
        h_prd, _ = self.prd_gru(prd_x)
        # h_usr = self.layernorm(h_usr)
        # h_prd = self.layernorm(h_prd)
        usr_x = self.usr_att(usr, h_usr, mask=mask)
        prd_x = self.prd_att(prd, h_prd, mask=mask)
        return usr_x, prd_x

class WordLevelRNN_UoP(nn.Module):
    def __init__(self, config):
        super().__init__()
        words_dim = config.words_dim
        usr_dim = config.usr_dim
        prd_dim = config.prd_dim
        word_hidden_dim = config.word_hidden_dim
        self.usr_gru = nn.LSTM(words_dim, word_hidden_dim // 2, bidirectional=True, batch_first=True)
        self.prd_gru = nn.LSTM(words_dim, word_hidden_dim // 2, bidirectional=True, batch_first=True)

        self.usr_att = UoPAttention(usr_dim, word_hidden_dim, config)
        # self.usr_att = BilinearAttention(usr_dim, word_hidden_dim, config)
        # self.prd_att = UoPAttention(prd_dim, word_hidden_dim, config)
        self.prd_att = BilinearAttention(prd_dim, word_hidden_dim, config)
        self.layernorm = nn.LayerNorm(word_hidden_dim)

    def forward(self, text, usr, prd, mask=None):
        if not hasattr(self, '_flattened'):
            self.usr_gru.flatten_parameters()
            self.prd_gru.flatten_parameters()
            setattr(self, '_flattened', True)
        h_usr, _ = self.usr_gru(text)
        h_prd, _ = self.prd_gru(text)
        # h_usr = self.layernorm(h_usr)
        # h_prd = self.layernorm(h_prd)
        usr_x = self.usr_att(usr, h_usr, mask=mask)
        prd_x = self.prd_att(prd, h_prd, mask=mask)
        return usr_x, prd_x