from flask import Flask, request, jsonify, render_template
import logging
import random
import numpy as np
from pymongo import MongoClient
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import load_model
import pickle

app = Flask(__name__)
logging.basicConfig(filename='app.log', level=logging.ERROR)

# Connect to MongoDB
uri = "mongodb+srv://Visitor:researchgogogo@reseearch.a2rwr6l.mongodb.net/"
client = MongoClient(uri)
db = client["research"]
intents_collection = db["data"]

# Load intents data from MongoDB
intents_data = list(intents_collection.find())
labels = list(set(intent['tag'] for intent in intents_data))

# Load tokenizer
tokenizer = Tokenizer()
tokenizer_path = 'tokenizer.pkl'
tokenizer = pickle.load(open(tokenizer_path, 'rb'))

# Load the trained Bi-LSTM model
model_path = 'bi_lstm_model'
bi_lstm = load_model(model_path)

def generate_bot_response(user_input, tokenizer, labels):
    sequences = tokenizer.texts_to_sequences([user_input])
    padded_sequences = pad_sequences(sequences, maxlen=max_len)
    predicted_class = np.argmax(bi_lstm.predict(padded_sequences), axis=1)[0]
    predicted_label = labels[predicted_class]
    responses = [intent['responses'] for intent in intents_data if intent['tag'] == predicted_label]
    bot_response = random.choice(responses)
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
        bot_response = generate_bot_response(user_input, tokenizer, labels)
        
        return jsonify({'bot_response': bot_response})

    except Exception as e:
        error_message = f"An error occurred: {str(e)}"
        logging.error(error_message)
        return jsonify({'bot_response': 'Internal Server Error'}), 500

if __name__ == '__main__':
    app.run(debug=True)
