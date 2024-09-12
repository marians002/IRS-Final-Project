import os
import cornac
import pandas as pd
from cornac.datasets import movielens
from cornac.models import ItemKNN, MF


class Hybrid(cornac.models.Recommender):
    def __init__(self, models, weights, name="Hybrid"):
        super().__init__(name=name)
        self.models = models
        self.weights = weights

    def fit(self, train_set, eval_set=None):
        super().fit(train_set, eval_set)
        for m in self.models:
            m.fit(train_set, eval_set)

    def score(self, user_idx, item_idx=None):
        ponderated_sum = 0
        # if each weight is 1/len(weights) is the average
        for idx in range(len(self.models)):
            ponderated_sum += self.models[idx].score(user_idx, item_idx) * self.weights[idx]

        return ponderated_sum / sum(self.weights)


class KFN(cornac.models.Recommender):
    def __init__(self, name="KFN"):
        super().__init__(name)

    def fit(self, train_set, eval_set=None):
        super().fit(train_set, eval_set)

    def score(self, user_idx, item_idx=None):
        if item_idx:
            return

    def recommend(self, user_id, k=-1, remove_seen=False, train_set=None):
        return super().recommend(user_id, k, remove_seen, train_set)


class DHybrid(Hybrid):
    def __init__(self, models, weights, name="Hybrid"):
        super().__init__(models, weights, name)
        self.kfn = KFN("KFN")

    def fit(self, train_set, eval_set=None):
        super().fit(train_set, eval_set)
        self.kfn.fit(train_set, eval_set)

    def score(self, user_idx, item_idx=None):
        return super().score(user_idx, item_idx)

    def recommend(self, user_id, k=-1, remove_seen=False, train_set=None, n=3):
        recommendations = super().recommend(user_id, k, remove_seen, train_set)
        return recommendations


def bayesian_avg(ratings, c, m):
    bay_avg = (c * m + ratings.sum()) / (c + ratings.count())
    return round(bay_avg, 3)


# Function to retrieve item names for recommended item IDs
def get_item_names(recommended_item_ids, dataset):
    # Create a mapping from item IDs to item names
    return [dataset[dataset['movie_id'] == int(item_id)]['title'].values[0] for item_id in recommended_item_ids]


def get_recommender(user):
    # Load the MovieLens 100K dataset
    data = movielens.load_feedback()
    dataset = cornac.data.Dataset.from_uir(data)

    # Convert the data to a pandas DataFrame
    df = pd.DataFrame(data, columns=['user_id', 'item_id', 'rating'])

    # Calculate count and mean
    df_stats = df
    df_stats['count'] = df_stats.groupby('user_id')['rating'].transform('count')
    df_stats['mean'] = df_stats.groupby('user_id')['rating'].transform('mean')

    movie_stats = df_stats.groupby('item_id')['rating'].agg(['count', 'mean'])

    c = movie_stats['count'].mean()
    m = movie_stats['mean'].mean()

    df_stats = df_stats.drop(columns='user_id')

    bayes_avg = df_stats.groupby('item_id')['rating'].agg(bayesian_avg, c, m).reset_index()
    bayes_avg.columns = ['item_id', 'bayesian_avg']
    df_stats = df_stats.merge(bayes_avg, on='item_id')
    df_stats = df_stats.drop_duplicates(subset='item_id')
    df_stats = df_stats.sort_values('bayesian_avg', ascending=False)

    mf = MF(k=10, max_iter=25, learning_rate=0.01, lambda_reg=0.02)
    svd = cornac.models.SVD()
    knn = ItemKNN(k=20, similarity='cosine', name='ItemKNN')
    bpr = cornac.models.BPR(k=10, max_iter=25, learning_rate=0.01, lambda_reg=0.02)
    hybrid = DHybrid([svd, bpr, knn], (6, 3, 1))

    hybrid.fit(dataset)

    # Obtain item recommendations for user
    recs = hybrid.recommend(user_id=str(user), k=5)

    script_dir = os.path.dirname(__file__)

    # Construct the absolute path to the sample_names.txt file
    file_path = os.path.join(script_dir, 'datasets/ml-100k/u.data')

    dataset = pd.read_csv(file_path, sep='\t', header=None, encoding='latin-1')
    dataset.columns = ['user_id', 'item_id', 'rating', 'timestamp']

    file_path = os.path.join(script_dir, 'datasets/ml-100k/u.item')
    movies = pd.read_csv(file_path, sep='|', header=None, encoding='latin-1')

    movies.columns = ['movie_id', 'title', 'release_date', 'video_release_date', 'IMDb_URL', 'unknown', 'Action', 'Adventure', 'Animation', 'Children', 'Comedy', 'Crime', 'Documentary', 'Drama', 'Fantasy', 'Film-Noir', 'Horror', 'Musical', 'Mystery', 'Romance', 'Sci-Fi', 'Thriller', 'War', 'Western']

    item_names = get_item_names(recs, movies)
    return item_names
