import cornac
import pandas as pd
from utils import load_movies, print_user_info, get_item_names
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
        train_set_to_dataset = cornac.data.Dataset.from_uirt(train_set)
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

    def recommend(self, user_id, k=-1, remove_seen=False, train_set=None, n=2):
        recommendations = super().recommend(user_id, k, remove_seen, train_set)
        if self.flag:
            sim_scores = self.sim_movies[int(recommendations[0])]
            similar_movies = [i[0] for i in sim_scores[:(n + 1)]]
            recommendations = recommendations + [str(i) for i in similar_movies]
        return recommendations


def get_recommender(user, dataset, verbose=False):
    """
    Generates movie recommendations for a given user using a hybrid recommender system.

    Args:
        user (str): The user ID for whom recommendations are to be generated.
        dataset (list): The dataset containing user ratings.
        verbose (bool, optional): If True, prints detailed user information. Defaults to False.

    Returns:
        list: A combined list of original and unsimilar movie recommendations.
    """
    # Load the movie dataset
    movies = load_movies()

    # Extract genre columns and compute cosine similarities between movies
    genre_columns = movies.columns[5:]
    movie_genres = movies[genre_columns]
    movie_similarities = cosine_similarity(movie_genres, movie_genres)

    # Create a dictionary of unsimilar movies based on similarity scores
    unsimilar_movies = {idx: sorted(enumerate(sim), key=lambda x: x[1], reverse=False)
                        for idx, sim in enumerate(movie_similarities)}

    # Initialize recommender models
    svd = cornac.models.SVD()
    knn = ItemKNN(k=20, similarity='cosine')
    bpr = cornac.models.BPR(k=10, max_iter=25, learning_rate=0.01, lambda_reg=0.02)

    # Create a hybrid recommender system
    hybrid = DHybrid([svd, bpr, knn], (4, 1, 6), flag=True, similar_movies=unsimilar_movies)

    # Fit the hybrid model to the dataset
    hybrid.fit(dataset)

    # Number of original and unsimilar recommendations to generate
    k = 5
    n = 3

    # Generate recommendations for the user
    recs = hybrid.recommend(user_id=str(user), k=k, n=n)

    # Retrieve names of original and unsimilar recommended movies
    original_recs = get_item_names(recs[:k], movies)
    diverse_movies = get_item_names(recs[k:], movies)
    original_and_diverse_movies = original_recs + diverse_movies

    # Print detailed user information if verbose is True
    if verbose:
        print_user_info(user, dataset, original_recs, diverse_movies, original_and_diverse_movies, movies)

    return original_and_diverse_movies