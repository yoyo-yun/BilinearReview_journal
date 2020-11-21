import torch
import torch.nn as nn
import math
import torch.nn.init as init


class SelfAttention(nn.Module):
    def __init__(self, input_dim, config):
        super(SelfAttention, self).__init__()
        self.pre_pooling_linear = nn.Linear(input_dim, config.pre_pooling_dim)
        self.pooling_linear = nn.Linear(config.pre_pooling_dim, 1)
        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)

    def forward(self, x, mask=None):
        weights = self.pooling_linear(torch.tanh(self.pre_pooling_linear(x))).squeeze(dim=2)
        if mask is not None:
            weights = weights.masked_fill(mask == 0, -1e9)
        weights = nn.Softmax(dim=-1)(weights)
        weights = self.dropout(weights)
        # return torch.mul(x, weights.unsqueeze(2)).sum(dim=1)
        return weights


class Bilinear(nn.Module):
    def __init__(self, input1_dim, input2_dim):
        super(Bilinear, self).__init__()
        self.bilinear_weights = nn.Parameter(torch.rand(input1_dim, input2_dim))

        bound = 1 / math.sqrt(self.bilinear_weights.size(1))
        init.uniform_(self.bilinear_weights, -bound, bound)
        # self.bilinear_weights.data.uniform_(-0.25, 0.25)

    def forward(self, input_1, input2):
        x = torch.matmul(input_1, self.bilinear_weights)
        if len(x.size()) != len(input2.size()):
            x = x.unsqueeze(1)
        return torch.tanh(torch.mul(x, input2))


class BilinearAttention(nn.Module):
    def __init__(self, input1_dim, input2_dim, config):
        super(BilinearAttention, self).__init__()
        self.bilinear = Bilinear(input1_dim, input2_dim)
        self.att = SelfAttention(input2_dim, config)

    def forward(self, input_1, input_2, mask=None):
        # input_1: usr or prd representation, input_2: text representation
        b_x = self.bilinear(input_1, input_2)
        att = self.att(b_x, mask=mask)
        output = torch.mul(input_2, att.unsqueeze(2)).sum(dim=1) + torch.mul(b_x, att.unsqueeze(2)).sum(dim=1)
        return output, att


class UoPAttention(nn.Module):
    def __init__(self, input1_dim, input2_dim, config):
        super(UoPAttention, self).__init__()
        self.pre_pooling_linear_attr = nn.Linear(input1_dim, config.pre_pooling_dim)
        self.pre_pooling_linear_text = nn.Linear(input2_dim, config.pre_pooling_dim)
        self.pooling_linear = nn.Linear(config.pre_pooling_dim, 1)

    def forward(self, input_1, input_2, mask=None):
        weights = self.pooling_linear(torch.tanh(self.pre_pooling_linear_attr(input_1).unsqueeze(1) + self.pre_pooling_linear_text(input_2))).squeeze(dim=2)
        if mask is not None:
            weights = weights.masked_fill(mask == 0, -1e9)
        weights = nn.Softmax(dim=-1)(weights)

        return torch.mul(input_2, weights.unsqueeze(2)).sum(dim=1), weights

if __name__ == '__main__':
    print("====testing bilinear modual...")
    bilinear = Bilinear(100, 32)
    a = torch.randn(64, 100)  # (bs, dim1)
    b = torch.randn(64, 15, 32)  # (bs, seq, dim2)
    output = bilinear(a,b)
    print(output.shape)

    c = torch.randn(64, 15, 100) # (bs, seq, dim1)
    d = torch.randn(64, 15, 32)  # (bs, seq, dim2)
    output1 = bilinear(c,d)
    print(output1.shape)

    e = torch.randn(64, 100) # (bs, dim1)
    f = torch.randn(64, 32)  # (bs, dim2)
    output2 = bilinear(e,f)
    print(output2.shape)
    print("done!")
    print()

    print("====testing bilinear attention modual...")
    from easydict import EasyDict as edict
    config = edict()
    config.pre_pooling_dim = 50
    bilinear_att = BilinearAttention(100, 32, config)
    a = torch.randn(64, 100)  # (bs, dim1)
    b = torch.randn(64, 15, 32)  # (bs, seq, dim2)
    output_att = bilinear_att(a,b)
    print(output_att.shape)
    print("done!")
    print()


