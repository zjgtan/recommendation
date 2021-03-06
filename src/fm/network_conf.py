# coding: utf8
from torch import nn
import torch

class Net(nn.Module):
    """深度因子分解网络
    """
    def __init__(self, num_user_id, num_item_id):
        """初始化
        """
        user_emb_size = 100
        item_emb_size = 100

        super(Net, self).__init__()
        # user_id的Embedding层
        self.user_id_embedding_layer = nn.Embedding(num_user_id + 1, user_emb_size)
        self.user_id_basis = nn.Embedding(num_user_id + 1, 1)
        # item_id的Embedding层
        self.item_id_embedding_layer = nn.Embedding(num_item_id + 1, item_emb_size)
        self.item_id_basis = nn.Embedding(num_item_id + 1, 1)

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
