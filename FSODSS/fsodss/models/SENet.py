# -*- coding: UTF-8 -*-
"""
SE structure
"""

import torch.nn as nn
import torch.nn.functional as F


class SELayer(nn.Module):

    def __init__(self, in_chnls, ratio):  # resnet中ratio取16
        super(SELayer, self).__init__()
        self.squeeze = nn.AdaptiveAvgPool2d((1, 1))
        input_se = int(in_chnls // ratio)
        self.compress = nn.Conv2d(in_chnls, input_se, 1, 1, 0)
        self.excitation = nn.Conv2d(input_se, in_chnls, 1, 1, 0)

    def forward(self, x):
        out = self.squeeze(x)
        out = self.compress(out)
        out = F.relu(out)
        out = self.excitation(out)
        return x*F.sigmoid(out)