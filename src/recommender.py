import pandas as pd
import numpy as np
import cornac
from cornac.datasets import movielens
from cornac.models import ItemKNN, MF

class Hybrid(cornac.models.Recommender):
    def __init__(self, models, weights, name="Hybrid"):
        super().__init__(name=name)
        self.models = models
        self.weights = weights

    def fit(self, train_set, eval_set):
        super().fit(train_set,eval_set)
        for m in self.models:
            m.fit(train_set, eval_set)

    def score(self, user_idx, item_idx=None):
        ponderated_sum = 0
        # if each weight is 1/len(weights) is the average
        for idx in range(len(self.models)):
            ponderated_sum += self.models[idx].score(user_idx, item_idx)*self.weights[idx]

        return ponderated_sum/sum(self.weights)
    
class KFN(cornac.models.Recommender):
    def __init__(self, name="KFN"):
        super().__init__(name)
        
    def fit(self, train_set, val_set=None):
        super().fit(train_set, val_set)
    def score(self, user_idx, item_idx=None):
        return 1
    def recommend(self, user_id, k=-1, remove_seen=False, train_set=None):
        return super().recommend(user_id, k, remove_seen, train_set)
    
class DHybrid(Hybrid):
    def __init__(self, models, weights, name="Hybrid"):
        super().__init__(models, weights, name)
        self.kfn = KFN("KFN")
        
    def fit(self, train_set, eval_set):
        super().fit(train_set, eval_set)
        self.kfn.fit(train_set, eval_set)
        
    def score(self, user_idx, item_idx=None):
        return super().score(user_idx, item_idx)
    
    def recommend(self, user_id, k=-1, remove_seen=False, train_set=None, n = 3):
        recommendations = super().recommend(user_id, k, remove_seen, train_set)
        if n < len(recommendations):
            recommendations[-n:] = self.kfn.recommend[:n]
        return recommendations
    

def bayesian_avg(ratings, c, m):
    bay_avg = (c * m + ratings.sum()) / (c + ratings.count())
    return round(bay_avg, 3)


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
    hybrid = DHybrid([svd, bpr, knn], (6,3,1))


    hybrid.fit(dataset)

    # Obtain item recommendations for user
    recs = hybrid.recommend(user_id=str(user), k=3)
    print(recs)
    return recs
