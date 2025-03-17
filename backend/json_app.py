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

app = Flask(__name__)
CORS(app, resources={r"/ask": {"origins": "http://localhost:3000"}})
logging.basicConfig(filename='app.log', level=logging.ERROR)

# exact paths to model, vectorizer and data
model_path = os.path.join(os.path.dirname(__file__), 'model.pkl')
vectorizer_path = os.path.join(os.path.dirname(__file__), 'vectorizer.pkl')
data_path = os.path.join(os.path.dirname(__file__), 'data.json')

# Load intents from data.json
with open(data_path, 'r') as file:
    intents = json.load(file)

# x and y 
# x - questions y - intents ( tag )
X = []
y = []
for intent in intents['intents']:
    for pattern in intent['patterns']:
        X.append(pattern)
        y.append(intent['tag'])

# vectorize using TFI-DF
vectorizer = TfidfVectorizer()
X_tfidf = vectorizer.fit_transform(X)

# encode label back into list
labels = list(set(y))
#create mapping for the patterns and tag 
label_to_idx = {label: idx for idx, label in enumerate(labels)}
# interates to list to map to its label and convert to numpy array
y_numeric = np.array([label_to_idx[label] for label in y])

#GridSearchCV for best parameter 
param_grid = {
    'C': [0.1, 1, 10],
    'kernel': ['linear', 'rbf', 'sigmoid'],
    'gamma': ['scale', 'auto'] + [0.001, 0.01, 0.1, 1, 10]
}
#model overview
svm_model = SVC()
grid_search = GridSearchCV(svm_model, param_grid, cv=5, n_jobs=-1)
grid_search.fit(X_tfidf, y_numeric)
model = grid_search.best_estimator_

# generate bot response using user input labels and the treshold for response similarity 
def generate_bot_response(user_input, labels, threshold=0.5):
    bot_name = "Faculty"
    # takes in user input and vectorize it before the model able to classify it 
    user_input_tfidf = vectorizer.transform([user_input])
    # it computes the similarity of user input and data
    similarities = cosine_similarity(user_input_tfidf, X_tfidf)
    # find the index with the most similar data and retrive it
    most_similar_idx = np.argmax(similarities)
    max_similarity = similarities[0, most_similar_idx]

    if max_similarity < threshold:
        # if the input is lower then the treshold it will give a given set of message
        bot_response = f"{bot_name}: Sorry, I didn't understand that."
    else:
        # the model predict the user input 
        predicted_label_numeric = model.predict(user_input_tfidf)
        predicted_label = labels[predicted_label_numeric[0]]

        # find the the response based on the predicted label 
        responses = []
        for intent_data in intents['intents']:
            if intent_data['tag'] == predicted_label:
                responses = intent_data['responses']
                break
        bot_response = random.choice(responses)
    return bot_response

#home
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
        bot_response = generate_bot_response(user_input, labels, threshold=0.8)
        return jsonify({'bot_response': bot_response})

    except Exception as e:
        error_message = f"An error occurred: {str(e)}"
        logging.error(error_message)  # Log error to app.log
        return jsonify({'bot_response': 'Internal Server Error'}), 500

if __name__ == '__main__':
    app.run(debug=True)
