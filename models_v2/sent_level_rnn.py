import torch
import torch.nn as nn
from models_v2.bilinear_att import SelfAttention, BilinearAttention, Bilinear, UoPAttention
from models_v2.interaction import Fusion


class SentLevelRNN(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.inter_sent_level = config.inter_sent_level
        self.shard_sent_level = config.shard_sent_level
        sentence_hidden_dim = config.sent_hidden_dim
        word_hidden_dim = config.word_hidden_dim
        usr_dim = config.usr_dim
        prd_dim = config.prd_dim
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

        if config.shard_sent_level == True:
            self.share_gru = nn.LSTM(word_hidden_dim, sentence_hidden_dim // 2, bidirectional=True, batch_first=True)
            self.register_parameter('self_gru', None)
            self.register_parameter('usr_gru', None)
            self.register_parameter('prd_gru', None)
        else:
            self.register_parameter('share_gru', None)
            self.self_gru = nn.LSTM(word_hidden_dim, sentence_hidden_dim // 2, bidirectional=True, batch_first=True)
            self.usr_gru = nn.LSTM(word_hidden_dim, sentence_hidden_dim // 2, bidirectional=True, batch_first=True)
            self.prd_gru = nn.LSTM(word_hidden_dim, sentence_hidden_dim // 2, bidirectional=True, batch_first=True)
        if self.inter_sent_level:
            self.concate = Fusion(sentence_hidden_dim, sentence_hidden_dim, sentence_hidden_dim)
        else:
            self.register_parameter('concate', None)

        self.usr_att = BilinearAttention(usr_dim, sentence_hidden_dim, config)
        # self.usr_att = UoPAttention(usr_dim, sentence_hidden_dim, config)
        # self.usr_text = Bilinear(usr_dim, sentence_hidden_dim)
        self.prd_att = BilinearAttention(prd_dim, sentence_hidden_dim, config)
        # self.prd_att = UoPAttention(prd_dim, sentence_hidden_dim, config)
        self.prd_text = Bilinear(prd_dim, sentence_hidden_dim)
        self.self_att = SelfAttention(sentence_hidden_dim, config)

    def forward(self, text_x, usr_x, prd_x, usr, prd, mask=None):
        if not hasattr(self, '_flattened'):
            self.self_gru.flatten_parameters()
            self.usr_gru.flatten_parameters()
            self.prd_gru.flatten_parameters()
            setattr(self, '_flattened', True)
        if self.shard_sent_level:
            h_text, _ = self.share_gru(self.dropout(text_x))
            h_usr, _ = self.share_gru(self.dropout(usr_x))
            h_prd, _ = self.share_gru(self.dropout(prd_x))
        else:
            h_text, _ = self.self_gru(self.dropout(text_x))
            h_usr, _ = self.usr_gru(self.dropout(usr_x))
            h_prd, _ = self.prd_gru(self.dropout(prd_x))
        weights = self.self_att(h_text, mask=mask)
        text_x = torch.mul(h_text, weights.unsqueeze(2)).sum(dim=1)
        usr_x, usr_att = self.usr_att(usr, h_usr, mask=mask)
        prd_x, prd_att = self.prd_att(prd, h_prd, mask=mask)
        if self.inter_sent_level:
            output = self.concate(usr_x, prd_x, text_x)
            return output
        else:
            return (text_x, weights), (usr_x, usr_att), (prd_x, prd_att)
