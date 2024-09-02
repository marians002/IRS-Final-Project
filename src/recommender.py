import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import cornac
from cornac.datasets import movielens
from cornac.eval_methods import RatioSplit
from cornac.metrics import Recall, Precision, MAE, RMSE
from cornac.models import ItemKNN, BPR, MF, SVD
from cornac.models import Recommender
from surprise import SVD
from cornac.eval_methods import RatioSplit

def bayesian_avg(ratings):
    bayesian_avg = (c*m+ratings.sum())/(c+ratings.count())
    return round(bayesian_avg, 3)

class Hybrid(cornac.models.Recommender):
    def __init__(self, mf_model, knn_model, name="Hybrid"):
        super().__init__(name=name)
        self.mf_model = mf_model
        self.knn_model = knn_model

    def fit(self, train_set, eval_set):
        super().fit(train_set,eval_set)
        self.mf_model.fit(train_set, eval_set)
        self.knn_model.fit(train_set, eval_set)

    def score(self, user_idx, item_idx=None, fraction=(10, 1)):
        mf_scores = self.mf_model.score(user_idx, item_idx)
        knn_scores = self.knn_model.score(user_idx, item_idx)
        total = sum(fraction)
        return (mf_scores*fraction[0] + knn_scores*fraction[1])/total
    
    data = movielens.load_feedback()


url = "http://files.grouplens.org/datasets/movielens/ml-100k/u.data"
# Load the MovieLens 100K dataset
data = movielens.load_feedback()
metadata = movielens.load_plot()

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

bayes_avg = df_stats.groupby('item_id')['rating'].agg(bayesian_avg).reset_index()
bayes_avg.columns = ['item_id', 'bayesian_avg']
df_stats = df_stats.merge(bayes_avg, on='item_id')
df_stats = df_stats.drop_duplicates(subset='item_id')
df_stats = df_stats.sort_values('bayesian_avg', ascending=False)

mf = MF(k=10, max_iter=25, learning_rate=0.01, lambda_reg=0.02)
svd = cornac.models.SVD()
knn = ItemKNN(k=20, similarity='cosine', name='ItemKNN')
bpr = cornac.models.BPR(k=10, max_iter=25, learning_rate=0.01, lambda_reg=0.02)
hybrid = Hybrid(mf_model=mf, knn_model=knn)



hybrid.fit(data)


# Obtain item recommendations for user U1
recs = hybrid.recommend(user_id="U1", k=5)
print(recs)
