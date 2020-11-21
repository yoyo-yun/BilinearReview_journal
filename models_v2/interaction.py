import math
import torch
import torch.nn as nn
import torch.nn.init as init
from functools import reduce


class Bilinear(nn.Module):
    def __init__(self, input1_dim, input2_dim, bias=True):
        super(Bilinear, self).__init__()
        self.bilinear_weights = nn.Parameter(torch.rand(input1_dim, input2_dim))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(input2_dim))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()


    def forward(self, input_1, input2):
        x = torch.matmul(input_1, self.bilinear_weights)
        if len(x.size()) != len(input2.size()):
            x = x.unsqueeze(1)
        output = torch.mul(x, input2)
        if self.bias is not None:
            output += self.bias
        return output

    def reset_parameters(self):
        init.kaiming_uniform_(self.bilinear_weights, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.bilinear_weights)
            bound = 1 / math.sqrt(fan_in)
            init.uniform_(self.bias, -bound, bound)



class Concate(nn.Module):
    def __init__(self, input1_dim, input2_dim, bias=True):
        super(Concate, self).__init__()
        self.weight_1 = nn.Parameter(torch.rand(input1_dim, input1_dim))
        self.weight_2 = nn.Parameter(torch.rand(input2_dim, input2_dim))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(input2_dim))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()


    def reset_parameters(self):
        init.kaiming_uniform_(self.weight_1, a=math.sqrt(5))
        init.kaiming_uniform_(self.weight_2, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.bilinear_weights)
            bound = 1 / math.sqrt(fan_in)
            init.uniform_(self.bias, -bound, bound)

    def forward(self, input_1, input_2):
        output = torch.add(torch.matmul(input_1, self.weight_1), torch.matmul(input_2, self.weight_2))
        if self.bias is not None:
            output += self.bias
        return output



class Fusion(nn.Module):
    def __init__(self, *input_dims, bias=True):
        super(Fusion, self).__init__()
        self.n_input = len(input_dims)
        self.input_dim = input_dims[0]
        self.weights = nn.ParameterList([nn.Parameter(torch.rand(self.input_dim, self.input_dim)) for _ in range(self.n_input)])
        if bias:
            self.bias = nn.Parameter(torch.Tensor(self.input_dim))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()


    def reset_parameters(self):
        for weight in self.weights:
            init.kaiming_uniform_(weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.weights[0])
            bound = 1 / math.sqrt(fan_in)
            init.uniform_(self.bias, -bound, bound)

    def forward(self, *inputs):
        output = [torch.matmul(i, weight) for i, weight in zip(inputs, self.weights)]
        output = reduce(lambda x, y: x + y, output)
        if self.bias is not None:
            output += self.bias
        return output


class CollaborativeInteraction(nn.Module):
    def __init__(self, *input_dims, config):
        super(CollaborativeInteraction, self).__init__()
        self.attribute_dim = input_dims[0]
        self.output = self.attribute_dim
        self.interaction = Bilinear(self.attribute_dim, self.attribute_dim)
        self.interaction_1 = Bilinear(self.attribute_dim, self.attribute_dim)
        self.activation = torch.nn.ReLU()

    def forward(self, *attrs):
        up = self.interaction(attrs[0], attrs[1])
        pu = self.interaction_1(attrs[1], attrs[0])
        output = torch.cat([up, pu], dim=-1)
        return output
