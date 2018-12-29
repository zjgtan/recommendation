# coding: utf8
import sys
import torch
import random
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

class ML1M(object):
    """movielens 1m数据集
    """
    def __init__(self, data_dir):
        """
        Args:
            data_dir: string, movielens 1m数据集目录
        """
        self.data_dir = data_dir
        self.load_dataset()
        self.n_samples = len(self.samples)

    def load_dataset(self):
        """加载数据集
        """
        # 用户画像
        self.users = self.load_user_data()
        # 电影画像
        self.movies = self.load_movie_data()
        # 打分数据, 电影的热度统计
        self.ratings = self.load_rating_data()
        # 生成训练样本
        self.samples = self.build_samples()

    def load_user_data(self):
        """加载用户画像
        schema: UserId, Gender, Age, Occupation
        """
        gender_dict = {"F": 0, "M": 1}
        age_dict = {"1": 0, "18": 1, "25": 2, "35": 3, "45": 4, "50": 5, "56": 6}

        users = {}
        for line in file(self.data_dir + "/users.dat"):
            toks = line.rstrip().split("::")
            users[toks[0]] = {}
            users[toks[0]]["gender"] = gender_dict[toks[1]]
            users[toks[0]]["age"] = age_dict[toks[2]]
            users[toks[0]]["occupation"] = int(toks[3])

        return users

    def load_movie_data(self):
        """加载电影画像
        schema: MovieId, Title, Genre
        """
        genres = ["Action",
            "Adventure",
            "Animation", 
            "Children's",
            "Comedy",
            "Crime",
            "Documentary",
            "Drama",
            "Fantasy",
            "Film-Noir",
            "Horror",
            "Musical",
            "Mystery",
            "Romance",
            "Sci-Fi",
            "Thriller",
            "War",
            "Western"]

        movies = {}
        for line in file(self.data_dir + "/movies.dat"):
            toks = line.rstrip().split("::")
            movies[toks[0]] = {}
            movies[toks[0]]["genre"] = []
            for elem in toks[2].split("|"):
                movies[toks[0]]["genre"].append(genres.index(elem) + 1)

            movies[toks[0]]["title"] = toks[1].split("(")[0].strip()

        return movies

    def load_rating_data(self):
        """打分数据
        """
        ratings = []
        for line in file(self.data_dir + "/ratings.dat"):
            toks = line.rstrip().split("::")
            ratings.append([toks[0], toks[1], int(toks[2])])
            
        return ratings

    def build_samples(self):
        """进行1:1的负采样
        """
        samples = []
        for uid, iid, score in self.ratings:
            record = {}
            record["iid"] = int(iid)
            record["uid"] = int(uid)
            record["score"] = score
            record["category"] = self.movies[iid]["genre"]
            record["age"] = self.users[uid]["age"]
            record["gender"] = self.users[uid]["gender"]
            record["occupation"] = self.users[uid]["occupation"]
            samples.append(record)

        return samples

    def make_batch_records(self, records):
        """转化为tensor
        """
        records["uid"] = torch.LongTensor(records["uid"])
        records["iid"] = torch.LongTensor(records["iid"])
        records["score"] = torch.FloatTensor(records["score"])
        records["gender"] = torch.LongTensor(records["gender"])
        records["age"] = torch.LongTensor(records["age"])
        records["occupation"] = torch.LongTensor(records["occupation"])
        records["category"] = [torch.LongTensor(category) for category in records["category"]]
        records["category"] = torch.nn.utils.rnn.pad_sequence(records["category"], batch_first = True)

        return records

if __name__ == "__main__":
    a = torch.LongTensor([1])
    b = torch.LongTensor([1, 2])
    print torch.nn.utils.rnn.pad_sequence([a, b], batch_first = True)
    dataset = ML1MExplicit("../../ml-1m")
    dataset.load_dataset()
    for record in dataset.next_batch(10):
        print record
