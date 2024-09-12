import matplotlib.pyplot as plt
import cornac
import pandas as pd
from cornac.datasets import movielens
from cornac.eval_methods import RatioSplit
from cornac.metrics import Recall, Precision, MAE, RMSE
from cornac.models import ItemKNN, BPR, MF
from cornac.eval_methods import RatioSplit
from sklearn.metrics.pairwise import cosine_similarity
import os

# Cargar la información de las películas desde el archivo 'u.item'
script_dir = os.path.dirname(__file__)

# Construct the absolute path to the sample_names.txt file
file_path = os.path.join(script_dir, 'datasets/ml-100k/u.data')
# utility = pd.read_csv(file_path, sep='\t', header=None, encoding='ascii')
# utility.columns = ['user_id', 'item_id', 'rating', 'timestamp']

# # Clean the rating column
# def clean_rating(rating):
#     try:
#         return float(rating)
#     except ValueError:
#         return None

# utility['rating'] = utility['rating'].apply(clean_rating)
# utility = utility.dropna(subset=['rating'])


dataset = cornac.datasets.movielens.load_feedback(fmt='UIRT')
# train_set = cornac.data.Dataset.from_uir(dataset)

file_path = os.path.join(script_dir, 'datasets/ml-100k/u.item')
movies = pd.read_csv(file_path, sep='|', header=None, encoding='latin-1')

movies.columns = ['movie_id', 'title', 'release_date', 'video_release_date', 'IMDb_URL', 'unknown', 'Action', 'Adventure', 'Animation', 'Children', 'Comedy', 'Crime', 'Documentary', 'Drama', 'Fantasy', 'Film-Noir', 'Horror', 'Musical', 'Mystery', 'Romance', 'Sci-Fi', 'Thriller', 'War', 'Western']

genre_columns = ['unknown', 'Action', 'Adventure', 'Animation', 'Children', 'Comedy', 'Crime', 'Documentary', 'Drama', 'Fantasy', 'Film-Noir', 'Horror', 'Musical', 'Mystery', 'Romance', 'Sci-Fi', 'Thriller', 'War', 'Western']

movie_genres = movies[genre_columns]
movie_similarities = cosine_similarity(movie_genres, movie_genres)
print(f"Dimensions of our genres cosine similarity matrix: {movie_similarities.shape}")

sim_movies = {}
for idx in range(len(movie_similarities)):
    sim_scores = list(enumerate(movie_similarities[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_movies[idx] = sim_scores

def user_rated_movies(user_id, train_set):
    rated_movies = []
    for row in train_set:
        if row[0] == user_id:
            rated_movies.append((row[0], row[1], row[3]))
    
    idx = min(rated_movies, key=lambda x: x[2])
    return int(idx[1])

# Example usage:
# user_id = 1
# rated_movies = user_rated_movies(user_id, train_set)
# print(rated_movies)

class Hybrid(cornac.models.Recommender):
    def __init__(self, models, weights, name="Hybrid"):
        super().__init__(name=name)
        self.models = models
        self.weights = weights

    def fit(self, train_set, eval_set=None):
        print(type(train_set))
        super().fit(train_set, eval_set)
        for m in self.models:
            m.fit(train_set, eval_set)

    def score(self, user_idx, item_idx=None):
        ponderated_sum = 0
        # if each weight is 1/len(weights)then the score is the average  
        for idx in range(len(self.models)):
            ponderated_sum += self.models[idx].score(user_idx, item_idx) * self.weights[idx]

        return ponderated_sum / sum(self.weights)

class DHybrid(Hybrid):
    def __init__(self, models, weights, name="Hybrid"):
        super().__init__(models, weights, name)
        
    def fit(self, train_set, eval_set=None):
        self.miscositas = train_set
        train_set_to_dataset = cornac.data.Dataset.from_uir(train_set)
        print('Fit')
        super().fit(train_set_to_dataset, eval_set)
        
    def score(self, user_idx, item_idx=None):
        return super().score(user_idx, item_idx)
    
    def recommend(self, user_id, k=-1, remove_seen=False, train_set=None, n=3):
        print('recommend')
        recommendations = super().recommend(user_id, k, remove_seen, train_set)
        idx = user_rated_movies(user_id, self.miscositas)
        sim_scores = sim_movies[idx]
        similar_movies = [i[0] for i in sim_scores[:5]]
        return recommendations.extend(similar_movies)

svd = cornac.models.SVD()
knn = ItemKNN(k=20, similarity='cosine', name='ItemKNN')
bpr = cornac.models.BPR(k=10, max_iter=25, learning_rate=0.01, lambda_reg=0.02)
hybrid = DHybrid([svd, bpr, knn], (6, 3, 1))

hybrid.fit(dataset, eval_set=None)

recs = hybrid.recommend(user_id='3', k=5)
