from flask import Flask, request, jsonify, render_template
from recommender import get_recommender
from map_names import map_names_to_ids

app = Flask(__name__)

# Global variable to store the name-to-ID mapping
name_to_id = None


def initialize_name_to_id():
    """
    Initializes the global name-to-ID mapping if it is not already initialized.
    """
    global name_to_id
    if name_to_id is None:
        name_to_id = map_names_to_ids()


def insert_to_database(user_name, user_ratings):
    pass


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
    user_id = name_to_id[user_name]
    if not user_id:
        print("Not valid user name")
        return jsonify({"error": "No user provided or user not found"}), 400
    recommendations = get_recommender(user_id, True)  # Call your recommendation function
    return jsonify(recommendations)


@app.route('/add_user', methods=['POST'])
def add_user():
    data = request.json
    user_name = data.get('user_name')
    user_ratings = data.get('user_ratings')

    if not user_name or not user_ratings:
        return jsonify({"message": "Invalid input"}), 400

    insert_to_database(user_name, user_ratings)
    return jsonify({"message": "User added"})


if __name__ == '__main__':
    initialize_name_to_id()  # Initialize the name-to-ID mapping when the program runs
    app.run(debug=True)
