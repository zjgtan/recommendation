# coding: utf8
from torch import nn
import torch

class AvgEmbeddingNet(nn.Module):
    def __init__(self, item_count, emb_size):
        super(AvgEmbeddingNet, self).__init__()

        self.emb_layer = nn.Embedding(item_count, emb_size)

    def forward(self, sequence):
        emb = self.emb_layer(sequence)
        emb = emb.permute(0, 2, 1)
        output = nn.AvgPool1d(emb.size()[-1], count_include_pad = False)(emb).view(emb.size()[0], -1)

        return output

class ItemFeatureNet(nn.Module):
    def __init__(self, num_item_id):
        super(ItemFeatureNet, self).__init__()

        self.item_id_embedding_layer = nn.Sequential(
            nn.Embedding(num_item_id + 1, 32),
            nn.Linear(32, 32),
            nn.Sigmoid())

        self.category_layer = AvgEmbeddingNet(19, 16)

        self.item_feature_combine_layer = nn.Sequential(
            nn.Linear(32 + 16, 200),
            nn.Tanh())

    def forward(self, batch_record):
        iid = self.item_id_embedding_layer(batch_record["iid"])
        category = self.category_layer(batch_record["category"])

        item_feature = self.item_feature_combine_layer(
                torch.cat([iid, category], 1))

        return item_feature

class UserFeatureNet(nn.Module):
    def __init__(self, num_user_id):
        """初始化
        """
        super(UserFeatureNet, self).__init__()

        user_fc_size = 0
        # user_id的Embedding层
        self.user_id_embedding_layer = nn.Sequential(
            nn.Embedding(num_user_id + 1, 32),
            nn.Linear(32, 32),
            nn.Sigmoid())

        user_fc_size += 32

        # user 性别Embedding层
        self.gender_embedding_layer = nn.Sequential(
            nn.Embedding(2, 16),
            nn.Linear(16, 16),
            nn.Sigmoid())

        user_fc_size += 16

        # user 职业Embedding层
        self.occupation_embedding_layer = nn.Sequential(
            nn.Embedding(21, 16),
            nn.Linear(16, 16),
            nn.Sigmoid())

        user_fc_size += 16

        # user 年龄Embedding层
        self.age_embedding_layer = nn.Sequential(
            nn.Embedding(7, 16),
            nn.Linear(16, 16),
            nn.Sigmoid())

        user_fc_size += 16

        self.user_feature_combine_layer = nn.Sequential(
            nn.Linear(user_fc_size, 200),
            nn.Tanh())

    def forward(self, batch_record):
        uid = self.user_id_embedding_layer(batch_record["uid"])
        gender = self.gender_embedding_layer(batch_record["gender"])
        occupation = self.occupation_embedding_layer(batch_record["occupation"])
        age = self.age_embedding_layer(batch_record["age"])

        user_feature = self.user_feature_combine_layer(
                torch.cat([uid, gender, occupation, age], 1))

        return user_feature


class Net(nn.Module):
    """深度因子分解网络
    """
    def __init__(self, num_user_id, num_item_id):

        super(Net, self).__init__()

        self.user_feature_layer = UserFeatureNet(num_user_id)
        self.item_feature_layer = ItemFeatureNet(num_item_id)

        self.neural_cf_layers = nn.Sequential(
                nn.Linear(400, 200),
                nn.Sigmoid(),
                nn.Linear(200, 200),
                nn.Sigmoid())

        self.output_layer = nn.Linear(200, 1)

    def forward(self, batch_record):
        """前向计算
        Args:
            batch_record: 
        """
        user_feature = self.user_feature_layer(batch_record)
        item_feature = self.item_feature_layer(batch_record)
        embedding_feature = torch.cat([user_feature, item_feature], 1)
        output = self.output_layer(self.neural_cf_layers(embedding_feature))
        output = output.view(1, -1)
        
        return output

if __name__ == "__main__":
    net = AvgEmbeddingNet(10, 10)
    print net(torch.LongTensor([[1, 2], [1,2]]))

    record = {"uid": torch.LongTensor([1, 2]), 
            "iid": torch.LongTensor([1, 2]), 
            "category": torch.LongTensor([[1], [2]]),
            "age": torch.LongTensor([1, 0]),
            "occupation": torch.LongTensor([1, 0]),
            "gender": torch.LongTensor([1, 0])}
    obj = NCF(10, 10, 10)
    print obj.forward(record)
