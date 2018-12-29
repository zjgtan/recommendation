# coding: utf8

import sys
sys.path.append("../common")
sys.path.append("../metrics")

import torch
from torch import nn

from network_conf import Net
from model import Model
from rmse import RMSE

class WideAndDeepModel(Model):
    def __init__(self, num_user, num_item):
        super(WideAndDeepModel, self).__init__()
        
        self.net = Net(num_user, num_item)
        self.optimizer = torch.optim.Adam(self.net.parameters())
        self.criterion = nn.MSELoss()
        self.metrics = {"rmse": RMSE()}
        

