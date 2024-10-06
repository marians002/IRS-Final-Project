from flask import Flask, request, jsonify, render_template
from recommender import get_recommender
from utils import map_names_to_ids, get_movie_id, load_movies, load_dataset
import pandas as pd
import os
from cornac.data import Reader

app = Flask(__name__)

# Global variable to store the name-to-ID mapping
name_to_id = {}


def add_user_rating(user_id, movie_id, rating):
    """
    Adds a new user rating to the dataset.

    Args:
        user_id (int): The ID of the user.
        movie_id (int): The ID of the movie.
        rating (int): The rating given by the user.

    Returns:
        Response: JSON response with a success or error message.
    """
    modified_dataset = load_dataset()
    # Add the new rating to the dataset
    timestamp = int(pd.Timestamp.now().timestamp())
    user_ratings = [rating for rating in modified_dataset if rating[0] == str(user_id)]
    movie_ids_list = [rating[1] for rating in user_ratings]

    print(sorted(movie_ids_list))
    print(type(movie_id))

    if str(movie_id) in movie_ids_list:
        return jsonify({"message": "This movie has already been rated by the user"}), 400
    new_user_ratings = [(str(user_id), str(movie_id), rating, timestamp)]

    # Append new user ratings to the dataset
    for rating in new_user_ratings:
        modified_dataset.append(rating)

    dataset_path = os.path.join(os.path.dirname(__file__), 'datasets/ml-100k/u.data')
    df = pd.DataFrame(modified_dataset, columns=['user_id', 'item_id', 'rating', 'timestamp'])
    df.to_csv(dataset_path, sep='\t', header=False, index=False)

    return jsonify({"message": "User rating added"})


def delete_user_rating(user_id, movie_id):
    """
    Deletes a user rating from the modified dataset.

    Args:
        user_id (int): The ID of the user.
        movie_id (int): The ID of the movie.

    Returns:
        Response: JSON response with a success or error message.
    """
    modified_dataset = load_dataset()
    # Find the rating to delete
    user_ratings = [rating for rating in modified_dataset if rating[0] == str(user_id)]
    movie_ids_list = [rating[1] for rating in user_ratings]

    if str(movie_id) not in movie_ids_list:
        return jsonify({"message": "This movie has not been rated by the user"}), 400

    # Remove the rating from the dataset
    modified_dataset = [rating for rating in modified_dataset if
                        not (rating[0] == str(user_id) and rating[1] == str(movie_id))]

    dataset_path = os.path.join(os.path.dirname(__file__), 'datasets/ml-100k/u.data')
    df = pd.DataFrame(modified_dataset, columns=['user_id', 'item_id', 'rating', 'timestamp'])
    df.to_csv(dataset_path, sep='\t', header=False, index=False)

    return jsonify({"message": "User rating deleted"})


def initialize_name_to_id():
    """
    Initializes the global name-to-ID mapping if it is not already initialized.
    """
    global name_to_id
    if not name_to_id:
        name_to_id = map_names_to_ids()


@app.route('/', methods=['GET'])
def index():
    """
    Renders the frontend HTML page.

    Returns:
        str: Rendered HTML template for the frontend.
    """
    return render_template('frontend.html')


@app.route('/recommend', methods=['POST'])
def recommend():
    """
    Handles the recommendation request. Expects a JSON payload with a 'user' field.

    Returns:
        Response: JSON response containing recommendations or an error message.
    """
    initialize_name_to_id()  # Ensure the name-to-ID mapping is initialized
    data = request.json
    user_name = data.get('user')
    user_id = name_to_id.get(user_name)
    if user_id is None:
        print("Not valid user name")
        return jsonify({"error": "No user provided or user not found"}), 400

    dataset_path = os.path.join(os.path.dirname(__file__), 'datasets/ml-100k/u.data')
    dataset = Reader().read(dataset_path, sep='\t', fmt='UIRT')
    recommendations = get_recommender(user_id, dataset, True)  # Call your recommendation function
    return jsonify(recommendations)


@app.route('/add_user', methods=['POST'])
def add_user():
    """
    Adds a new user rating to the dataset.

    This endpoint expects a JSON payload with the following fields:
    - user_name (str): The name of the user.
    - movie_name (str): The name of the movie.
    - movie_rating (int): The rating given by the user (must be between 1 and 5).

    Returns:
        Response: JSON response with a success or error message.
    """
    data = request.json
    user_name = data.get('user_name')
    movie_name = data.get('movie_name')
    movie_rating = data.get('movie_rating')

    # Check if the request contains the required fields
    if not user_name:
        return jsonify({"message": "Not a valid user"}), 400
    if not movie_name:
        return jsonify({"message": "Not a valid movie"}), 400
    if not movie_rating:
        return jsonify({"message": "Not a valid rating"}), 400

    # Check data-types in the entries:
    if not isinstance(user_name, str):
        return jsonify({"message": "User name must be a string"}), 400
    if not isinstance(movie_name, str):
        return jsonify({"message": "Movie name must be a string"}), 400
    # Try to convert the movie rating to an integer:
    try:
        movie_rating = int(movie_rating)
    except ValueError:
        return jsonify({"message": "Movie rating must be a number"}), 400
    if not (1 <= movie_rating <= 5):
        return jsonify({"message": "Movie rating must be between 1 and 5"}), 400

    if not name_to_id:
        initialize_name_to_id()

    # Check if the user already exists
    if user_name in name_to_id:
        user_id = name_to_id[user_name]
    else:
        user_id = max(name_to_id.values(), default=0) + 1
        name_to_id[user_name] = user_id
        # Modify sample_names.txt to save the changes
        file_path = os.path.join(os.path.dirname(__file__), 'sample_names.txt')
        with open(file_path, 'a') as f:
            f.write(f"\n{user_name}")

    # Load the movies dataset
    movies = load_movies()

    movie_id = get_movie_id(movie_name, movies)
    if movie_id is None:
        return jsonify({"message": "Movie not found"}), 400

    return add_user_rating(user_id, movie_id, movie_rating)


@app.route('/delete_rating', methods=['POST'])
def delete_rating():
    """
    Deletes a user rating from the dataset.

    This endpoint expects a JSON payload with the following fields:
    - user_name (str): The name of the user.
    - movie_name (str): The name of the movie.

    Returns:
        Response: JSON response with a success or error message.
    """
    data = request.json
    user_name = data.get('user_name')
    movie_name = data.get('movie_name')

    if not user_name:
        return jsonify({"message": "Not a valid user"}), 400
    if not movie_name:
        return jsonify({"message": "Not a valid movie"}), 400

    if not name_to_id:
        initialize_name_to_id()

    user_id = name_to_id.get(user_name)
    if user_id is None:
        return jsonify({"message": "User not found"}), 400

    # Load the movies dataset
    movies = load_movies()

    movie_id = get_movie_id(movie_name, movies)
    if movie_id is None:
        return jsonify({"message": "Movie not found"}), 400

    return delete_user_rating(user_id, movie_id)


if __name__ == '__main__':
    initialize_name_to_id()  # Initialize the name-to-ID mapping when the program runs
    app.run(debug=True)
