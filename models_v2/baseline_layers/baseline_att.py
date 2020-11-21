import math
import torch
import torch.nn as nn
import torch.nn.init as init
from functools import reduce

class SelfAttention(nn.Module):
    def __init__(self, input_dim, config):
        super(SelfAttention, self).__init__()
        self.pre_pooling_linear = nn.Linear(input_dim, config.pre_pooling_dim)
        self.pooling_linear = nn.Linear(config.pre_pooling_dim, 1)

    def forward(self, x, mask=None):
        weights = self.pooling_linear(torch.tanh(self.pre_pooling_linear(x))).squeeze(dim=2)
        if mask is not None:
            weights = weights.masked_fill(mask == 0, -1e9)
        weights = nn.Softmax(dim=-1)(weights)

        return torch.mul(x, weights.unsqueeze(2)).sum(dim=1)


class ConcateAttention(nn.Module):
    def __init__(self, config, input_dim, usr_dim, prd_dim, bias=True):
        super(ConcateAttention, self).__init__()
        # self.weights = nn.ParameterList([nn.Parameter(torch.rand(attr_dim, config.pre_pooling_dim)) for attr_dim in attrs_dim])
        self.weight_usr = nn.Parameter(torch.rand(usr_dim, config.pre_pooling_dim))
        self.weight_prd = nn.Parameter(torch.rand(prd_dim, config.pre_pooling_dim))
        self.weight_repr = nn.Parameter(torch.rand(input_dim, config.pre_pooling_dim))
        if bias:
            self.bias = nn.Parameter(torch.rand(config.pre_pooling_dim))
        else:
            self.register_parameter('bias', None)
        self.weight_vector = nn.Linear(config.pre_pooling_dim, 1)
        self.reset_parameters()

    def reset_parameters(self):
        # for weight in self.weights:
        #     init.kaiming_uniform_(weight, a=math.sqrt(5))
        init.kaiming_uniform_(self.weight_usr, a=math.sqrt(5))
        init.kaiming_uniform_(self.weight_prd, a=math.sqrt(5))
        init.kaiming_uniform_(self.weight_repr, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight_repr)
            bound = 1 / math.sqrt(fan_in)
            init.uniform_(self.bias, -bound, bound)

    def forward(self, input_repr, usr, prd, mask=None):
        # output_attrs = [torch.matmul(input_attr, weight) for weight, input_attr in zip(self.weights, input_attrs)]
        # output_attr = reduce(lambda x,y: x+y, output_attrs)
        output_usr = torch.matmul(usr, self.weight_usr)
        output_prd = torch.matmul(prd, self.weight_prd)
        output_repr = torch.matmul(input_repr, self.weight_repr)
        # output = output_usr + output_prd + output_repr
        output = output_usr + output_repr + output_prd
        if self.bias is not None:
            output += self.bias
        attention_score = self.weight_vector(torch.tanh(output)).squeeze(-1) # (bs, seq)
        if mask is not None:
            attention_score = attention_score.masked_fill(mask == 0, -1e9)
        attention_score = nn.Softmax(dim=1)(attention_score)
        return torch.mul(input_repr, attention_score.unsqueeze(2)).sum(dim=1) # (bs, input_dim)


class Bilinear(nn.Module):
    def __init__(self, input1_dim, input2_dim):
        super(Bilinear, self).__init__()
        self.bilinear_weights = nn.Parameter(torch.rand(input1_dim, input2_dim))
        self.bilinear_weights.data.uniform_(-0.25, 0.25)

    def forward(self, input_1, input2):
        x = torch.matmul(input_1, self.bilinear_weights)
        if len(x.size()) != len(input2.size()):
            x = x.unsqueeze(1)
        return torch.tanh(torch.mul(x, input2))


class BilinearAttention_v2(nn.Module):
    def __init__(self, input1_dim, input2_dim, config):
        super(BilinearAttention_v2, self).__init__()
        self.bilinear = Bilinear(input1_dim, input2_dim)
        self.att = SelfAttention(input2_dim, config)

    def forward(self, input_1, input_2, mask=None):
        # input_1: usr or prd representation, input_2: text representation
        b_x = self.bilinear(input_1, input_2)
        att = self.att(b_x, mask=mask)
        return att


class BilinearAttention(nn.Module):
    def __init__(self, input1_dim, input2_dim, config):
        super(BilinearAttention, self).__init__()
        self.bilinear = Bilinear(input1_dim, input2_dim)
        self.pre_pooling_linear = nn.Linear(input2_dim, config.pre_pooling_dim)
        self.pooling_linear = nn.Linear(config.pre_pooling_dim, 1)

    def forward(self, input_1, input_2, mask=None):
        # input_1: usr or prd representation, input_2: text representation
        b_x = self.bilinear(input_1, input_2)
        weights = self.pooling_linear(torch.tanh(self.pre_pooling_linear(b_x))).squeeze(dim=2)
        if mask is not None:
            weights = weights.masked_fill(mask == 0, -1e9)
        weights = nn.Softmax(dim=-1)(weights)
        # return torch.mul(input_2, weights.unsqueeze(2)).sum(dim=1)
        return (torch.mul(input_2, weights.unsqueeze(2)).sum(dim=1) + torch.mul(b_x, weights.unsqueeze(2)).sum(dim=1)) / 2.


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

        return torch.mul(input_2, weights.unsqueeze(2)).sum(dim=1)


class UoPConcatAttention(nn.Module):
    def __init__(self, attr_dim, input_dim, config, bias=True):
        super(UoPConcatAttention, self).__init__()
        # self.weights = nn.ParameterList([nn.Parameter(torch.rand(attr_dim, config.pre_pooling_dim)) for attr_dim in attrs_dim])
        self.weight_attr = nn.Parameter(torch.rand(attr_dim, config.pre_pooling_dim))
        self.weight_repr = nn.Parameter(torch.rand(input_dim, config.pre_pooling_dim))
        if bias:
            self.bias = nn.Parameter(torch.rand(config.pre_pooling_dim))
        else:
            self.register_parameter('bias', None)
        self.weight_vector = nn.Linear(config.pre_pooling_dim, 1)
        self.reset_parameters()

    def reset_parameters(self):
        # for weight in self.weights:
        #     init.kaiming_uniform_(weight, a=math.sqrt(5))
        init.kaiming_uniform_(self.weight_attr, a=math.sqrt(5))
        init.kaiming_uniform_(self.weight_repr, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight_repr)
            bound = 1 / math.sqrt(fan_in)
            init.uniform_(self.bias, -bound, bound)

    def forward(self, attr, input_repr, mask=None):
        # output_attrs = [torch.matmul(input_attr, weight) for weight, input_attr in zip(self.weights, input_attrs)]
        # output_attr = reduce(lambda x,y: x+y, output_attrs)
        output_attr = torch.matmul(attr, self.weight_attr).unsqueeze(1)
        output_repr = torch.matmul(input_repr, self.weight_repr)
        output = output_repr + output_attr
        if self.bias is not None:
            output += self.bias
        attention_score = self.weight_vector(torch.tanh(output)).squeeze(-1)  # (bs, seq)
        if mask is not None:
            attention_score = attention_score.masked_fill(mask == 0, -1e9)
        attention_score = nn.Softmax(dim=1)(attention_score)
        return torch.mul(input_repr, attention_score.unsqueeze(2)).sum(dim=1)  # (bs, input_dim)

