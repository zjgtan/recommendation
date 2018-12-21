# coding: utf8
from torch import nn
import torch

class DeepFM(nn.Module):
    """深度因子分解网络
    """
    def __init__(self, num_user_id, num_item_id):
        """初始化
        """
        super(DeepFM, self).__init__()
        # user_id的Embedding层
        self.user_id_embedding_layer = nn.Embedding(num_user_id + 1, 10)
        self.user_id_basis = nn.Embedding(num_user_id + 1, 1)
        # item_id的Embedding层
        self.item_id_embedding_layer = nn.Embedding(num_item_id + 1, 10)
        self.item_id_basis = nn.Embedding(num_item_id + 1, 1)
        # 全连接层
        '''
        self.fc_layer = nn.Sequential(
                nn.Linear(40, 20),
                nn.Sigmoid())
        # 输出层
        #self.output_layer = nn.Sequential(
        #        nn.Linear(200, 2),
        #        nn.LogSoftmax())

        self.output_layer = nn.Sequential(
                nn.Linear(20, 1))
        '''

    def forward(self, batch_record):
        """前向计算
        Args:
            batch_record: 
        """
        uids = batch_record["uid"]
        user_id_embedding = self.user_id_embedding_layer(uids)

        iids = batch_record["iid"]
        item_id_embedding = self.item_id_embedding_layer(iids)
        
        output = self.user_id_basis(uids) + \
                self.item_id_basis(iids) + \
                (user_id_embedding * item_id_embedding).sum(1).view(-1, 1)

        output = output.view(1,-1)
        return output

        '''
        user_item_feature = torch.cat([user_id_embedding, item_id_embedding], 1)
        fc = self.fc_layer(user_item_feature)
        output = self.output_layer(fc)

        return output
        '''

if __name__ == "__main__":
    record = {"uid": torch.LongTensor([1, 2]), "iid": torch.LongTensor([1, 2])}
    obj = DeepFM(10, 10)
    print obj.forward(record)
