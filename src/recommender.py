import os
import cornac
import pandas as pd
from cornac.models import ItemKNN, Recommender
from sklearn.metrics.pairwise import cosine_similarity


def bayesian_avg(ratings, c, m):
    """
    Computes the Bayesian average of ratings.
    
    Args:
        ratings (pd.Series): Series of ratings.
        c (float): Mean count of ratings.
        m (float): Mean rating.
    
    Returns:
        float: Bayesian average rating.
    """
    bay_avg = (c * m + ratings.sum()) / (c + ratings.count())
    return round(bay_avg, 3)


def get_item_names(recommended_item_ids, dataset):
    """
    Retrieves item names for the given recommended item IDs.

    Args:
        recommended_item_ids (list): List of recommended item IDs.
        dataset (pd.DataFrame): DataFrame containing item information.

    Returns:
        list: List of item names.
    """
    names = []  # Initialize an empty list to store the names of recommended items

    # Iterate over each movie_id in the dataset
    for item in dataset['movie_id']:
        # Iterate over each recommended item ID
        for item_id in recommended_item_ids:
            # Check if the current movie_id matches the recommended item ID
            if item == int(item_id):
                # Retrieve the title of the movie and append it to the names list
                name = dataset[dataset['movie_id'] == int(item_id)]['title'].values[0]
                names.append(name)

    return names  # Return the list of item names


class Hybrid(Recommender):
    """
    A hybrid recommender system that combines multiple models with specified weights.

    Attributes:
        models (list): List of recommender models to be combined.
        weights (tuple): List of weights corresponding to each model.
        name (str): Name of the hybrid model.
    """

    def __init__(self, models, weights, name="Hybrid"):
        """
        Initializes the Hybrid model with the given models and weights.

        Args:
            models (list): List of recommender models.
            weights (tuple): List of weights for each model.
            name (str): Name of the hybrid model.
        """
        super().__init__(name=name)
        self.models = models
        self.weights = weights

    def fit(self, train_set, eval_set=None):
        """
        Fits the hybrid model to the training data.

        Args:
            train_set: Training dataset.
            eval_set: Evaluation dataset (optional).
        """
        super().fit(train_set, eval_set)
        for m in self.models:
            m.fit(train_set, eval_set)

    def score(self, user_idx, item_idx=None):
        """
        Computes the score for a given user and item by combining the scores from all models.

        Args:
            user_idx: Index of the user.
            item_idx: Index of the item (optional).

        Returns:
            float: Combined score.
        """
        ponderated_sum = 0
        # if each weight is 1/len(weights) then the score is the average
        for idx in range(len(self.models)):
            ponderated_sum += self.models[idx].score(user_idx, item_idx) * self.weights[idx]

        return ponderated_sum / sum(self.weights)


class DHybrid(Hybrid):
    """
    A hybrid recommender system that includes a KFN model.

    Attributes:
        sim_movies (dict): Dictionary of similar movies.
    """

    def __init__(self, models, weights, name="Hybrid", flag=False, similar_movies=None):
        super().__init__(models, weights, name)
        self.flag = flag
        self.dataset = None
        self.sim_movies = similar_movies

    def fit(self, train_set, eval_set=None):
        """
        Fits the DHybrid model to the training data.

        Args:
            train_set: Training dataset.
            eval_set: Evaluation dataset (optional).
        """
        self.dataset = train_set
        train_set_to_dataset = cornac.data.Dataset.from_uir(train_set)
        super().fit(train_set_to_dataset, eval_set)

    def score(self, user_idx, item_idx=None):
        """
        Computes the score for a given user and item by combining the scores from all models.

        Args:
            user_idx: Index of the user.
            item_idx: Index of the item (optional).

        Returns:
            float: Combined score.
        """
        return super().score(user_idx, item_idx)

    def recommend(self, user_id, k=-1, remove_seen=False, train_set=None, n=3):
        recommendations = super().recommend(user_id, k, remove_seen, train_set)
        if self.flag:
            sim_scores = self.sim_movies[int(recommendations[0])]
            similar_movies = [i[0] for i in sim_scores[:n]]
            recommendations = recommendations + [str(i) for i in similar_movies]
        return recommendations


def get_recommender(user):
    """
    Generates item recommendations for a given user.

    Args:
        user (int): ID of the user.

    Returns:
        list: List of recommended item names.
    """
    # Load the MovieLens dataset
    dataset = cornac.datasets.movielens.load_feedback(fmt='UIRT')

    # Load movie information
    file_path = os.path.join(os.path.dirname(__file__), 'datasets/ml-100k/u.item')
    movies = pd.read_csv(file_path, sep='|', header=None, encoding='latin-1')
    movies.columns = ['movie_id', 'title', 'release_date', 'video_release_date', 'IMDb_URL', 'unknown', 'Action',
                      'Adventure', 'Animation', 'Children', 'Comedy', 'Crime', 'Documentary', 'Drama', 'Fantasy',
                      'Film-Noir', 'Horror', 'Musical', 'Mystery', 'Romance', 'Sci-Fi', 'Thriller', 'War', 'Western']

    # Compute movie similarities
    genre_columns = movies.columns[5:]
    movie_genres = movies[genre_columns]
    movie_similarities = cosine_similarity(movie_genres, movie_genres)

    # Create a dictionary of un-similar movies
    unsimilar_movies = {idx: sorted(enumerate(sim), key=lambda x: x[1], reverse=False)
                        for idx, sim in enumerate(movie_similarities)}

    # Initialize and fit the hybrid model
    svd = cornac.models.SVD()
    knn = ItemKNN(k=20, similarity='cosine')
    bpr = cornac.models.BPR(k=10, max_iter=25, learning_rate=0.01, lambda_reg=0.02)
    hybrid = DHybrid([svd, bpr, knn], (4, 1, 6), flag=True, similar_movies=unsimilar_movies)
    hybrid.fit(dataset)

    # Get recommendations for the user
    recs = hybrid.recommend(user_id=str(user), k=5)
    print(recs)
    return get_item_names(recs, movies)
