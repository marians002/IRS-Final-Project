from flask import Flask, request, jsonify, render_template
from recommender import get_recommender
from map_names import map_names_to_ids

app = Flask(__name__)

# Global variable to store the name-to-ID mapping
name_to_id = None


def initialize_name_to_id():
    global name_to_id
    if name_to_id is None:
        print('Here')
        name_to_id = map_names_to_ids()


@app.route('/', methods=['GET'])
def index():
    return render_template('frontend.html')


@app.route('/recommend', methods=['POST'])
def recommend():
    initialize_name_to_id()  # Ensure the name-to-ID mapping is initialized
    data = request.json
    user_name = data.get('user')
    user_id = name_to_id[user_name]
    if not user_id:
        print("Not valid user name")
        return jsonify({"error": "No user provided or user not found"}), 400
    recommendations = get_recommender(user_id)  # Call your recommendation function
    return jsonify(recommendations)


if __name__ == '__main__':
    initialize_name_to_id()  # Initialize the name-to-ID mapping when the program runs
    app.run(debug=True)
