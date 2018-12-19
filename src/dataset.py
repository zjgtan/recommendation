# coding: utf8
from sklearn.model_selection import train_test_split

class ML1M(object):
    """movielens 1m数据集
    """
    def __init__(self, data_dir):
        """
        Args:
            data_dir: string, movielens 1m数据集目录
        """
        self.data_dir = data_dir
    
    def load_dataset(self):
        """加载数据集
        """
        # 用户画像
        self.users = self.load_user_data()
        # 电影画像
        self.movies = self.load_movie_data()
        # 打分数据
        self.ratings = self.load_rating_data()

        X, y = self.build(self.ratings, 
                self.users, self.movies)

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3)

        return X_train, X_test, y_train, y_test

    def load_user_data(self):
        """
        """
    def build(self, ratings, users, movies):
        """
        Args:
            ratings: list, 用户打分数据
            users: dict, 用户画像
            movies: dict, 电影画像
        """

        return X, y

