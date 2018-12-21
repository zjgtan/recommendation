# coding: utf8
import sys
import torch
import random
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

class ML100Explicit(object):
    """movielens 1m数据集
    """
    def __init__(self, data_dir):
        """
        Args:
            data_dir: string, movielens 1m数据集目录
        """
        self.data_dir = data_dir

    def get_user_num(self):
        return max(map(int, self.users.keys()))

    def get_movie_num(self):
        return max(map(int, self.movies.keys()))
    
    def load_dataset(self):
        """加载数据集
        """
        # 用户画像
        #self.users = self.load_user_data()
        # 电影画像
        #self.movies = self.load_movie_data()
        # 打分数据, 电影的热度统计
        self.ratings = self.load_rating_data()
        # 负采样
        self.samples = self.build_samples()
        # 划分训练集和测试集
        self.train_set, self.test_set = train_test_split(self.samples, test_size = 0.3)

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
            users[toks[0]]["occupation"] = toks[3]

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
                movies[toks[0]]["genre"].append(genres.index(elem))

            movies[toks[0]]["title"] = toks[1].split("(")[0].strip()

        return movies

    def load_rating_data(self):
        """打分数据
        """
        ratings = []
        for line in file(self.data_dir + "/ua.base"):
            toks = line.rstrip().split("\t")
            ratings.append([toks[0], toks[1], int(toks[2])])
            
        return ratings

    def train_reader(self):
        """
        Args:
            ratings: list, 用户打分数据
            users: dict, 用户画像
            movies: dict, 电影画像
        """
        
        train_set = shuffle(self.train_set)
        #train_set = self.train_set

        for uid, iid, score in train_set:
            record = {}
            record["uid"] = int(uid)
            record["iid"] = int(iid)
            record["score"] = score
            yield record

    def test_reader(self):
        """
        Args:
            ratings: list, 用户打分数据
            users: dict, 用户画像
            movies: dict, 电影画像
        """
        for uid, iid, score in self.test_set:
            record = {}
            record["uid"] = int(uid)
            record["iid"] = int(iid)
            record["score"] = score
            yield record
    
    def next_batch(self, batch_size, is_train = True):
        count = 0
        records = {}
        if is_train == False:
            reader = self.test_reader
        else:
            reader = self.train_reader

        for record in reader:
            if count == batch_size:
                yield self.make_batch_records(records)
                records = {}
                count = 0

            count += 1

            for key, value in record.iteritems():
                 records.setdefault(key, [])
                 records[key].append(value)

        if len(records) != 0:
            yield self.make_batch_records(records)

    def next_epoch(self):
        return self.train_epoch_records

    def test_reader(self):
        """
        Args:
            ratings: list, 用户打分数据
            users: dict, 用户画像
            movies: dict, 电影画像
        """
        
        test_set = shuffle(self.test_set)

        for uid, iid, score in train_set:
            record = {}
            record["uid"] = int(uid)
            record["iid"] = int(iid)
            record["score"] = score
            yield record
    
    def next_test_batch(self):
        count = 0
        records = {}
        for record in self.reader():
            for key, value in record.iteritems():
                 records.setdefault(key, [])
                 records[key].append(value)

        return self.make_batch_records(records)


    def make_batch_records(self, records):
        """转化为tensor
        """
        records["uid"] = torch.LongTensor(records["uid"])
        records["iid"] = torch.LongTensor(records["iid"])
        #records["score"] = torch.LongTensor(records["score"])
        records["score"] = torch.FloatTensor(records["score"])

        return records


class ML1MExplicit(object):
    """movielens 1m数据集
    """
    def __init__(self, data_dir):
        """
        Args:
            data_dir: string, movielens 1m数据集目录
        """
        self.data_dir = data_dir

    def get_user_num(self):
        return max(map(int, self.users.keys()))

    def get_movie_num(self):
        return max(map(int, self.movies.keys()))
    
    def load_dataset(self):
        """加载数据集
        """
        # 用户画像
        self.users = self.load_user_data()
        # 电影画像
        self.movies = self.load_movie_data()
        # 打分数据, 电影的热度统计
        self.ratings = self.load_rating_data()
        # 负采样
        self.samples = self.build_samples()
        # 划分训练集和测试集
        self.train_set, self.test_set = train_test_split(self.samples, test_size = 0.3)

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
            users[toks[0]]["occupation"] = toks[3]

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
                movies[toks[0]]["genre"].append(genres.index(elem))

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

    def do_sample(self, count_list, num):
        sum_count = sum(count_list)

        # 计算累计分布函数
        acc_dist = [0]
        acc_count = 0
        for i in range(len(count_list)):
            acc_count = acc_count + count_list[i]
            acc_dist.append(acc_count * 1. / sum_count)

        # 采样 
        sample_indexes = []
        for _ in range(num):
            p = random.uniform(0, 1)
            # 二分查找
            left = 0
            right = len(acc_dist) - 1

            while left <= right:
                mid = (left + right) / 2
                # 条件
                if acc_dist[mid] >= p and acc_dist[mid - 1] < p:
                    sample_indexes.append(mid - 1)
                    break
                elif acc_dist[mid - 1] >= p:
                    right = mid - 1
                elif acc_dist[mid] < p:
                    left = mid + 1

        return sample_indexes

    def do_negative_sampling(self, items):
        """负采样
        """
        negative_items = list(set(self.movie_count_dict.keys()) - \
                set(items))
        negative_item_count_list = [self.movie_count_dict[item] for item in negative_items]
        indexes = self.do_sample(negative_item_count_list, len(items) * 2)
        sample_negative_items = [negative_items[index] for index in indexes]

        return sample_negative_items
        
    def build_samples(self):
        """进行1:1的负采样
        """
        samples = []
        for uid, item, score in self.ratings:
            #samples.append([uid, item, score * 1. / 5])
            samples.append([uid, item, score])

        return samples

    def train_reader(self):
        """
        Args:
            ratings: list, 用户打分数据
            users: dict, 用户画像
            movies: dict, 电影画像
        """
        
        train_set = shuffle(self.train_set)
        #train_set = self.train_set

        for uid, iid, score in train_set:
            record = {}
            record["uid"] = int(uid)
            record["iid"] = int(iid)
            record["score"] = score
            yield record

    def next_batch(self, batch_size, is_train = True):
        if is_train:
            reader = self.train_reader
        else:
            reader = self.test_reader

        count = 0
        records = {}
        for record in reader():
            if count == batch_size:
                yield self.make_batch_records(records)
                records = {}
                count = 0

            count += 1

            for key, value in record.iteritems():
                 records.setdefault(key, [])
                 records[key].append(value)

        if len(records) != 0:
            yield self.make_batch_records(records)

    def next_epoch(self):
        return self.train_epoch_records

    def test_reader(self):
        """
        Args:
            ratings: list, 用户打分数据
            users: dict, 用户画像
            movies: dict, 电影画像
        """
        
        for uid, iid, score in self.test_set:
            record = {}
            record["uid"] = int(uid)
            record["iid"] = int(iid)
            record["score"] = score
            yield record
    
    def next_test_batch(self):
        count = 0
        records = {}
        for record in self.reader():
            for key, value in record.iteritems():
                 records.setdefault(key, [])
                 records[key].append(value)

        return self.make_batch_records(records)


    def make_batch_records(self, records):
        """转化为tensor
        """
        records["uid"] = torch.LongTensor(records["uid"])
        records["iid"] = torch.LongTensor(records["iid"])
        #records["score"] = torch.LongTensor(records["score"])
        records["score"] = torch.FloatTensor(records["score"])

        return records


class ML1MImplicit(object):
    """movielens 1m数据集
    """
    def __init__(self, data_dir):
        """
        Args:
            data_dir: string, movielens 1m数据集目录
        """
        self.data_dir = data_dir

    def get_user_num(self):
        return max(map(int, self.users.keys()))

    def get_movie_num(self):
        return max(map(int, self.movies.keys()))
    
    def load_dataset(self):
        """加载数据集
        """
        # 用户画像
        self.users = self.load_user_data()
        # 电影画像
        self.movies = self.load_movie_data()
        # 打分数据, 电影的热度统计
        self.ratings, self.movie_count_dict = self.load_rating_data()
        # 负采样
        self.samples = self.build_samples()
        # 划分训练集和测试集
        self.train_set, self.test_set = train_test_split(self.samples, test_size = 0.3)

        self.train_epoch_records = [batch for batch in self.next_batch(len(self.train_set))][0]

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
            users[toks[0]]["occupation"] = toks[3]

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
                movies[toks[0]]["genre"].append(genres.index(elem))

            movies[toks[0]]["title"] = toks[1].split("(")[0].strip()

        return movies

    def load_rating_data(self):
        """打分数据
        """
        ratings = {}
        movie_count_dict = {}
        for line in file(self.data_dir + "/ratings.dat"):
            toks = line.rstrip().split("::")
            if int(toks[2]) <= 3: continue
            ratings.setdefault(toks[0], [])
            ratings[toks[0]].append(toks[1])

            movie_count_dict.setdefault(toks[1], 0)
            movie_count_dict[toks[1]] += 1
            
        return ratings, movie_count_dict

    def do_sample(self, count_list, num):
        sum_count = sum(count_list)

        # 计算累计分布函数
        acc_dist = [0]
        acc_count = 0
        for i in range(len(count_list)):
            acc_count = acc_count + count_list[i]
            acc_dist.append(acc_count * 1. / sum_count)

        # 采样 
        sample_indexes = []
        for _ in range(num):
            p = random.uniform(0, 1)
            # 二分查找
            left = 0
            right = len(acc_dist) - 1

            while left <= right:
                mid = (left + right) / 2
                # 条件
                if acc_dist[mid] >= p and acc_dist[mid - 1] < p:
                    sample_indexes.append(mid - 1)
                    break
                elif acc_dist[mid - 1] >= p:
                    right = mid - 1
                elif acc_dist[mid] < p:
                    left = mid + 1

        return sample_indexes

    def do_negative_sampling(self, items):
        """负采样
        """
        negative_items = list(set(self.movie_count_dict.keys()) - \
                set(items))
        negative_item_count_list = [self.movie_count_dict[item] for item in negative_items]
        indexes = self.do_sample(negative_item_count_list, len(items) * 2)
        sample_negative_items = [negative_items[index] for index in indexes]

        return sample_negative_items
        
    def build_samples(self):
        """进行1:1的负采样
        """
        samples = []
        for uid, items in self.ratings.iteritems():
            for item in items:
                samples.append([uid, item, 1])

            negative_items = self.do_negative_sampling(items)
            for item in negative_items:
                samples.append([uid, item, 0])

        return samples

    def reader(self):
        """
        Args:
            ratings: list, 用户打分数据
            users: dict, 用户画像
            movies: dict, 电影画像
        """
        
        #train_set = shuffle(self.train_set)[:1000]
        train_set = self.train_set

        for uid, iid, score in train_set:
            record = {}
            record["uid"] = int(uid)
            record["iid"] = int(iid)
            record["score"] = score
            yield record
    
    def next_batch(self, batch_size):
        count = 0
        records = {}
        for record in self.reader():
            if count == batch_size:
                yield self.make_batch_records(records)
                records = {}
                count = 0

            count += 1

            for key, value in record.iteritems():
                 records.setdefault(key, [])
                 records[key].append(value)

        if len(records) != 0:
            yield self.make_batch_records(records)

    def next_epoch(self):
        return self.train_epoch_records

    def test_reader(self):
        """
        Args:
            ratings: list, 用户打分数据
            users: dict, 用户画像
            movies: dict, 电影画像
        """
        
        test_set = shuffle(self.test_set)

        for uid, iid, score in train_set:
            record = {}
            record["uid"] = int(uid)
            record["iid"] = int(iid)
            record["score"] = score
            yield record
    
    def next_test_batch(self):
        count = 0
        records = {}
        for record in self.reader():
            for key, value in record.iteritems():
                 records.setdefault(key, [])
                 records[key].append(value)

        return self.make_batch_records(records)


    def make_batch_records(self, records):
        """转化为tensor
        """
        records["uid"] = torch.LongTensor(records["uid"])
        records["iid"] = torch.LongTensor(records["iid"])
        #records["score"] = torch.LongTensor(records["score"])
        records["score"] = torch.LongTensor(records["score"])

        return records


if __name__ == "__main__":
    obj = ML1M("../ml-1m")
    obj.load_dataset()

    for batch in obj.next_batch(20):
        print batch
        break
