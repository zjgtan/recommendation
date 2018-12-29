# coding: utf8
"""
Usage: 训练评估器
"""

from logzero import logger
import json

class Trainer(object):
    """训练Controller
    """
    def __init__(self, model):
        """初始化
        Args: 
            model: 注册模型
            dataset: 注册的数据集
            batch_size: 训练的batch_size
        """
        self.model = model

    def fit(self, reader, n_epoch):
        """
        Args:
            n_epoch: 迭代轮数
        """
        metrics_list = []
        for epoch in range(n_epoch):
            # 将模型设置为训练模式 
            self.model.train()
            for batch_id, batch_records in enumerate(reader.next_batch(is_shuffle = True)):
                preds = self.model.net(batch_records)
                scores = batch_records["score"]
                loss = self.model.criterion(preds, scores)

                if batch_id % 100  == 0:
                    logger.info("iter: %d, batch_id: %d, loss: %f" % (epoch, batch_id, loss.item()))

                self.model.optimizer.zero_grad()
                loss.backward()
                self.model.optimizer.step()

            metrics = self.evaluation(reader)
            logger.info("iter: %d, train metrics: %s" % (epoch, json.dumps(metrics)))

            metrics_list.append(metrics)

        return metrics_list

    def evaluation(self, reader):
        """评估迭代效果
        """
        # 将网络设置为预测模式
        self.model.net.eval()

        metrics = {}
        for key, metric in self.model.metrics.iteritems():
            metrics[key] = metric.eval(self.model, reader)

        return metrics

