import torch
import torch.nn as nn


class Classifier(nn.Module):
    def __init__(self,input_dim, config):
        super(Classifier, self).__init__()
        self.pre_classifier = nn.Linear(input_dim, config.pre_classifier_dim)
        self.classifier = nn.Linear(config.pre_classifier_dim, config.num_classes)

    def forward(self, hidden):
        pre_hidden = torch.tanh(self.pre_classifier(hidden))
        logits = self.classifier(pre_hidden)
        return logits



