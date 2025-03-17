from flask import Flask, request, jsonify, render_template
import os
import json
import random
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.metrics.pairwise import cosine_similarity
from flask_cors import CORS
import logging
from pymongo.mongo_client import MongoClient
from pymongo.server_api import ServerApi

app = Flask(__name__)
CORS(app, resources={r"/ask": {"origins": "http://localhost:3000"}})
logging.basicConfig(filename='app.log', level=logging.ERROR)

uri = "mongodb+srv://Visitor:researchgogogo@reseearch.a2rwr6l.mongodb.net/"
client = MongoClient(uri, server_api=ServerApi('1'))
db = client["research"]
intents_collection = db["data"]

# Load data from MongoDB
intents_data = list(intents_collection.find())
labels = list(set(intent['tag'] for intent in intents_data))
label_to_idx = {label: idx for idx, label in enumerate(labels)}

X = []
y = []
for intent in intents_data:
    for pattern in intent['patterns']:
        X.append(pattern)
        y.append(intent['tag'])

vectorizer = TfidfVectorizer()
X_tfidf = vectorizer.fit_transform(X)
y_numeric = np.array([label_to_idx[label] for label in y])

param_grid = {
    'C': [0.1, 1, 10],
    'kernel': ['linear', 'rbf', 'sigmoid'],
    'gamma': ['scale', 'auto'] + [0.001, 0.01, 0.1, 1, 10]
}

svm_model = SVC()
grid_search = GridSearchCV(svm_model, param_grid, cv=5, n_jobs=-1)
grid_search.fit(X_tfidf, y_numeric)
best_svm_model = grid_search.best_estimator_

def generate_bot_response(user_input, threshold=0.8):
    bot_name = "Faculty"
    
    user_input_tfidf = vectorizer.transform([user_input])
    similarities = cosine_similarity(user_input_tfidf, X_tfidf)
    most_similar_idx = np.argmax(similarities)
    max_similarity = similarities[0, most_similar_idx]

    if max_similarity < threshold:
        # No similar intents above the threshold, suggest the most similar pattern
        suggested_pattern = X[most_similar_idx]
        bot_response = f"{bot_name}: Did you mean: '{suggested_pattern}'? ('yes' or 'no')"
    else:
        # Use the best SVM model to predict the intent label
        predicted_label_numeric = best_svm_model.predict(user_input_tfidf)
        predicted_label = labels[predicted_label_numeric[0]]
        
        # Retrieve responses for the predicted intent
        responses = []
        for intent_data in intents_data:
            if intent_data['tag'] == predicted_label:
                responses = intent_data['responses']
                break
        
        if responses:
            bot_response = random.choice(responses)
        else:
            bot_response = f"{bot_name}: Sorry, I'm unable to respond right now."
    
    return bot_response


@app.route('/')
def home():
    return render_template('index.html')
    
@app.route('/ask', methods=['POST'])
def ask():
    try:
        data = request.get_json()
        if 'user_input' not in data:
            return jsonify({'bot_response': 'no input'}), 400

        user_input = data['user_input']
        bot_response = generate_bot_response(user_input, threshold=0.5)
        
        # Check if the bot suggested a pattern
        if "Did you mean" in bot_response:
            suggested_pattern = bot_response.split("Did you mean: '")[1].split("'?")[0]
            user_response = data.get('user_response', '').strip().lower()

            if user_response == 'yes' and suggested_pattern == 'IOD Codes':
                # Provide more information about 'IOD Codes'
                # This section should be updated based on your intent data structure
                responses = []
                for intent_data in intents_data:
                    if intent_data['tag'] == suggested_pattern:
                        responses = intent_data['responses']
                        break

                if responses:
                    bot_response = random.choice(responses)
                else:
                    bot_response = f"Sorry, I don't have information about '{suggested_pattern}' right now."
            else:
                bot_response = f"Please specify your question."

        return jsonify({'bot_response': bot_response})

    except KeyError:
        return jsonify({'bot_response': 'Invalid request data. "user_input" key missing.'}), 400

    except Exception as e:
        error_message = f"An error occurred: {str(e)}"
        logging.error(error_message)
        return jsonify({'bot_response': 'Internal Server Error'}), 500


if __name__ == '__main__':
    app.run(debug=True)
