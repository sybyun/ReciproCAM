from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn
import torch.nn.functional as F


class Cls_Header(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Cls_Header, self).__init__()

        intrim_channel = int(0.5*in_channels)
        #last_channel = 128
        self.fc1 = nn.Linear(in_channels, intrim_channel)
        self.bn1 = nn.BatchNorm1d(intrim_channel)
        self.relu1 = nn.ReLU()
        #self.fc2 = nn.Linear(intrim_channel, last_channel)
        #self.bn2 = nn.BatchNorm1d(last_channel)
        #self.relu2 = nn.ReLU()
        self.logit = nn.Linear(intrim_channel, out_channels)

    def forward(self, x):
        net = torch.flatten(x, 1)
        net = self.fc1(net)
        net = self.bn1(net)
        net = self.relu1(net)
        #net = self.fc2(net)
        #net = self.bn2(net)
        #net = self.relu2(net)
        net = self.logit(net)

        return net

