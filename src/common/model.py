# coding: utf8

class Model(object):
    def __init__(self):
        # 网络
        self.net = None
        # 优化方法
        self.optimizer = None
        # 损失函数
        self.criterion = None
        # 评估方法
        self.metrics = None

    def train(self):
        self.net.train()

    def eval(self):
        self.net.eval()

    def reset(self):
        self.net.reset_parameters()

        
            




        
