# coding: utf8

import sys
sys.path.append("../common")
sys.path.append("../metrics")
from model import Model
from network_conf import Net
from rmse import RMSE

import torch
from torch import nn

class FMModel(Model):
    def __init__(self, num_user, num_item):
        super(FMModel, self).__init__()

        self.net = Net(num_user, num_item)
        self.criterion = nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.net.parameters())

        self.metrics = {"rmse": RMSE()}

