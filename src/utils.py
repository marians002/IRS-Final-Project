import os
import re
import pandas as pd
from datetime import datetime
from cornac.data import Reader


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
                # modify this to append the object at the beggining of the list
                names.append(name)

    return names  # Return the list of item names


def print_user_info(user, dataset, original_recs, diverse_movies, original_and_diverse_movies, movies):
    """
    Prints user information including their dataset, original recommended movies,
    unsimilar movies added to recommendations, and combined list of original and unsimilar movies.

    Args:
        user (str): The user ID.
        dataset (list): The dataset containing user ratings.
        original_recs (list): List of original recommended movies.
        diverse_movies (list): List of unsimilar movies added to recommendations.
        original_and_diverse_movies (list): Combined list of original and unsimilar movies.
        movies (pd.DataFrame): DataFrame containing movie information.
    """
    user_ratings = [rating for rating in dataset if rating[0] == str(user)]
    movie_ids_list = [rating[1] for rating in user_ratings]
    favorite_movies = get_item_names(movie_ids_list, movies)
    print("User ", user, " dataset:")
    for name in favorite_movies:
        print(f"- {name}")

    print("\nOriginal recommended movies:")
    for name in original_recs:
        print(f"- {name}")

    print("\nUnsimilar movies added to recommendations:")
    for name in diverse_movies:
        print(f"- {name}")

    print("\nCombined list of original and unsimilar movies:")
    for name in original_and_diverse_movies:
        print(f"- {name}")


def map_names_to_ids(file_path=None):
    """
    Maps names to unique IDs from a file.

    Args:
        file_path (str, optional): The path to the file containing names.
                                   If None, defaults to 'sample_names.txt' in the current script's directory.

    Returns:
        dict: A dictionary mapping names to unique IDs.
    """
    if file_path is None:
        # Get the directory of the current script
        script_dir = os.path.dirname(__file__)
        # Construct the absolute path to the sample_names.txt file
        file_path = os.path.join(script_dir, 'sample_names.txt')

    with open(file_path, 'r') as file:
        names = file.readlines()

    # Remove any leading/trailing whitespace characters (like newlines)
    names = [name.strip() for name in names]

    # Create a dictionary mapping names to IDs (e.g., Angelica Powers: 1)
    name_to_id = {name: i + 1 for i, name in enumerate(names)}

    return name_to_id


def load_movies():
    """
    Loads the movie dataset from a file.

    Returns:
        pd.DataFrame: DataFrame containing movie information.
    """
    dataset_path = os.path.join(os.path.dirname(__file__), 'datasets/ml-100k/u.item')
    movies = pd.read_csv(dataset_path, sep='|',
                         encoding='latin-1', header=None,
                         names=['movie_id', 'title', 'release_date', 'video_release_date',
                                'IMDb_URL', 'unknown', 'Action', 'Adventure', 'Animation',
                                'Children', 'Comedy', 'Crime', 'Documentary', 'Drama', 'Fantasy',
                                'Film-Noir', 'Horror', 'Musical', 'Mystery', 'Romance', 'Sci-Fi',
                                'Thriller', 'War', 'Western'])
    return movies


def load_dataset():
    """
    Loads the user rating dataset from a file.

    Returns:
        list: List of user ratings in the format (user_id, item_id, rating, timestamp).
    """
    dataset_path = os.path.join(os.path.dirname(__file__), 'datasets/ml-100k/u.data')
    modified_dataset = Reader().read(dataset_path, sep='\t', fmt='UIRT')
    return modified_dataset


def normalize_title(title):
    """
    Normalizes a movie title by removing the year, converting to lowercase,
    and moving 'the' to the beginning if the title ends with ', The'.

    Args:
        title (str): The original movie title.

    Returns:
        str: The normalized movie title.
    """
    # Remove the year from the title
    title = re.sub(r'\s*\(\d{4}\)', '', title)
    # Convert the title to lowercase
    title = title.lower()
    # Move "the" to the beginning if the title ends with ", The"
    if title.endswith(', the'):
        title = 'the ' + title[:-5]
    return title


def get_movie_id(movie_name, dataset):
    """
    Retrieves the movie ID for a given movie name from the dataset.

    Args:
        movie_name (str): The name of the movie.
        dataset (pd.DataFrame): The dataset containing movie information.

    Returns:
        int: The ID of the movie.
    """
    # Apply the normalization function to the 'title' column
    dataset['title'] = dataset['title'].apply(normalize_title)

    # Normalize movie_name
    movie_name = normalize_title(movie_name)

    # Retrieve the movie_id for the given movie_name, if fails, return None
    if movie_name in dataset['title'].values:
        movie_id = dataset[dataset['title'] == movie_name]['movie_id'].values[0]
    else:
        movie_id = None
    return movie_id


def calculate_bayesian_average(v, R, m, C):
    """
    Calculate the Bayesian average rating for a movie.

    Args:
        v (int): Number of votes for the movie.
        R (float): Average rating of the movie.
        m (float): Minimum votes required to be listed in the Top Picks.
        C (float): Mean rating across all movies.

    Returns:
        float: Bayesian average rating.
    """
    return (v * R + m * C) / (v + m)


def get_top_picks(top_n=5):
    """
    Get the top picks movies based on Bayesian average rating.

    Args:
        top_n (int, optional): Number of top picks to return. Defaults to 5.

    Returns:
        list: List of top picks movie names.
    """
    # Read the u.item file
    df = load_movies()
    dataset_path = os.path.join(os.path.dirname(__file__), 'datasets/ml-100k/u.data')

    # Assuming we have a DataFrame `ratings` with columns ['user_id', 'movie_id', 'rating', 'timestamp']
    ratings = pd.read_csv(dataset_path, sep='\t', names=['user_id', 'movie_id', 'rating', 'timestamp'])

    # Calculate the mean rating across all movies
    C = ratings['rating'].mean()

    # Calculate the number of votes for each movie
    movie_stats = ratings.groupby('movie_id').agg({'rating': ['mean', 'count']})
    movie_stats.columns = ['mean_rating', 'vote_count']

    # Minimum votes required to be listed in the Top Picks
    m = movie_stats['vote_count'].quantile(0.90)

    # Filter out movies with less than m votes
    qualified_movies = movie_stats[movie_stats['vote_count'] >= m]

    # Calculate Bayesian average for each qualified movie
    qualified_movies['bayesian_avg'] = qualified_movies.apply(
        lambda x: calculate_bayesian_average(x['vote_count'], x['mean_rating'], m, C), axis=1)

    # Sort movies by Bayesian average
    top_picks = qualified_movies.sort_values(by='bayesian_avg', ascending=False).head(top_n)

    # Merge with the original movie titles
    top_picks = top_picks.merge(df[['movie_id', 'title']], on='movie_id')

    # Return only the movie names
    return top_picks['title'].tolist()


def get_latest_releases(top_n=5):
    """
    Get the latest released movies.

    Args:
        top_n (int, optional): Number of latest releases to return. Defaults to 5.

    Returns:
        list: List of latest released movie names.
    """
    # Read the u.item file
    df = load_movies()
    # Convert release_date to datetime
    df['release_date'] = pd.to_datetime(df['release_date'], format='%d-%b-%Y', errors='coerce')
    # Sort movies by release date
    latest_releases = df.sort_values(by='release_date', ascending=False).head(top_n)

    # Return only the movie titles
    return latest_releases['title'].tolist()