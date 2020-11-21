import torch
import torch.nn as nn

from models_v2.bilinear_att import Bilinear
from models_v2.classifier import Classifier
from models_v2.sent_level_rnn import SentLevelRNN
from models_v2.word_level_rnn import WordLevelRNN
from models_v2.interaction import CollaborativeInteraction


class Net(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.text_embedding = config.text_embedding
        self.usr_embedding = config.usr_embedding
        self.prd_embedding = config.prd_embedding
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        # self.text_embed = nn.Embedding.from_pretrained(self.text_embedding, freeze=True)
        # self.usr_embed = nn.Embedding.from_pretrained(self.usr_embedding, freeze=False)
        # self.prd_embed = nn.Embedding.from_pretrained(self.prd_embedding, freeze=False)
        self.text_embed = nn.Embedding(config.text_embedding.size(0), config.text_embedding.size(1),
                                       padding_idx=config.pad_idx)
        self.text_embed.weight.data.copy_(self.text_embedding)
        self.text_embed.weight.requires_grad = False
        # self.text_embed.weight.requires_grad = True

        self.usr_embed = nn.Embedding(config.usr_embedding.size(0), config.usr_dim,
                                      padding_idx=config.pad_idx)
        # self.usr_embed.weight.data.copy_(self.usr_embedding)
        self.usr_embed.weight.data.copy_(torch.Tensor(config.usr_embedding.size(0), config.usr_dim).uniform_(-0.01,0.01))
        self.usr_embed.weight.requires_grad = True

        self.prd_embed = nn.Embedding(config.prd_embedding.size(0), config.prd_dim,
                                      padding_idx=config.pad_idx)
        # self.prd_embed.weight.data.copy_(self.prd_embedding)
        self.prd_embed.weight.data.copy_(torch.Tensor(config.prd_embedding.size(0), config.prd_dim).uniform_(-0.01,0.01))
        self.prd_embed.weight.requires_grad = True

        self.word_attention_rnn = WordLevelRNN(config)
        self.sentence_attention_rnn = SentLevelRNN(config)
        self.interaction = CollaborativeInteraction(config.usr_dim, config.prd_dim, config=config)
        self.classifier_bup = Classifier(config.usr_dim+config.prd_dim, config)
        self.classifier_text = Classifier(config.sent_hidden_dim, config)
        self.classifier_usr = Classifier(config.sent_hidden_dim, config)
        self.classifier_prd = Classifier(config.sent_hidden_dim, config)
        self.classifier_sents = Classifier(config.sent_hidden_dim, config)
        if self.config.inter_sent_level:
            self.classifier_all = Classifier(config.usr_dim + config.sent_hidden_dim, config)
        else:
            self.classifier_all = Classifier(config.usr_dim + config.prd_dim + config.sent_hidden_dim * 3, config)

    def forward(self, text, usr, prd, mask=None):
        # text: (batch, sent, word)
        # usr: (batch, 1)
        # prd: (batch, 1)
        # mask: (batch, sent, word)
        # word embedding
        if len(text.size()) == 3:
            text = text.permute(1, 0, 2)  # text: (sent, batch, word)
            text = self.text_embed(text) # text: (sent, batch, word, dim)
        elif len(text.size()) > 3:
            text = text.permute(1, 0, 2, 3)  # text: (sent, batch, word, dim)
        else:
            print("Error Input...")
            exit()

        usr = usr.squeeze(-1)  # usr: (batch, )
        prd = prd.squeeze(-1)  # prd: (batch, )
        usr = self.usr_embed(usr)
        prd = self.prd_embed(prd)

        # collaborative interaction
        bup = self.interaction(usr, prd)

        # logits
        logits_bup = self.classifier_bup(bup)

        mask_word = None
        mask_sent = None
        if mask is not None:
            mask_word = mask.permute(1, 0, 2)  # text: (sent, batch, word)
            mask_sent = mask.long().sum(2) > 0  # (batch, sent)

        num_sentences = text.size(0)
        words_text = []
        words_usr = []
        words_prd = []
        words_text_atts = []
        words_usr_atts = []
        words_prd_atts = []
        for i in range(num_sentences):
            text_x, usr_x, prd_x = self.word_attention_rnn(text[i], usr, prd, mask=mask_word[i])
            words_text.append(text_x[0])
            words_usr.append(usr_x[0])
            words_prd.append(prd_x[0])
            words_text_atts.append(text_x[1])
            words_usr_atts.append(usr_x[1])
            words_prd_atts.append(prd_x[1])
        words_text = torch.stack(words_text, 1)  # (batch, sents, dim)
        words_usr = torch.stack(words_usr, 1)  # (batch, sents, dim)
        words_prd = torch.stack(words_prd, 1)  # (batch, sents, dim)
        if num_sentences == 1:
            words_text_atts = words_text_atts[0].unsqueeze(1)  # (batch, sentence, word)
            words_usr_atts = words_usr_atts[0].unsqueeze(1)
            words_prd_atts = words_prd_atts[0].unsqueeze(1)
        else:
            words_text_atts = torch.stack(words_text_atts, 1) # (batch, sentence, word)
            words_usr_atts = torch.stack(words_usr_atts, 1)
            words_prd_atts = torch.stack(words_prd_atts, 1)

        if self.config.inter_sent_level:
            sents = self.sentence_attention_rnn(words_text, words_usr, words_prd, usr, prd,
                                          mask=mask_sent)
            logits_sents = self.classifier_sents(sents)
            logits_all = self.classifier_all(torch.cat([bup, sents], dim=-1))
            return logits_bup, logits_sents, logits_all, logits_all, logits_all

        else:
            sents_text_x, sents_usr_x, sents_prd_x = self.sentence_attention_rnn(words_text, words_usr, words_prd, usr, prd,
                                                                       mask=mask_sent)
            sents_text = self.dropout(sents_text_x[0])
            sents_usr = self.dropout(sents_usr_x[0])
            sents_prd = self.dropout(sents_prd_x[0])
            sents_text_atts, sents_usr_atts, sents_prd_atts = sents_text_x[1], sents_usr_x[1], sents_prd_x[1]
            logits_usr = self.classifier_usr(sents_usr)
            logits_prd = self.classifier_prd(sents_prd)
            logits_text = self.classifier_text(sents_text)
            logits_all = self.classifier_all(torch.cat([bup, sents_usr, sents_prd, sents_text], dim=-1))

            return (logits_bup, logits_text, logits_prd, logits_usr, logits_all),\
                   (words_text_atts, words_usr_atts, words_prd_atts, sents_text_atts, sents_usr_atts, sents_prd_atts)
