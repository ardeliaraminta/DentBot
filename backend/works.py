from flask import Flask, request, jsonify, render_template
import os
import json
import random
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Bidirectional, LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from flask_cors import CORS
import logging
from sklearn.metrics import classification_report

app = Flask(__name__)
CORS(app, resources={r"/ask": {"origins": "http://localhost:3000"}})
logging.basicConfig(filename='app.log', level=logging.ERROR)

# Paths to model, vectorizer, and data
model_path = os.path.join(os.path.dirname(__file__), 'model.pkl')
vectorizer_path = os.path.join(os.path.dirname(__file__), 'vectorizer.pkl')
data_path = os.path.join(os.path.dirname(__file__), 'new.json')

# Load intents from data.json
with open(data_path, 'r') as file:
    intents = json.load(file)

# Extract questions (X) and intents (y)
X = []
y = []
for intent in intents['intents']:
    for pattern in intent['patterns']:
        X.append(pattern)
        y.append(intent['tag'])

# Tokenize text data
tokenizer = Tokenizer()
tokenizer.fit_on_texts(X)
X_sequences = tokenizer.texts_to_sequences(X)
max_len = max(len(seq) for seq in X_sequences)
X_padded = pad_sequences(X_sequences, maxlen=max_len)
labels = list(set(y))
label_to_idx = {label: idx for idx, label in enumerate(labels)}
y_numeric = np.array([label_to_idx[label] for label in y])

# split data into train and validation sets
X_train, X_val, y_train, y_val = train_test_split(X_padded, y_numeric, test_size=0.2, random_state=42)

vocab_size = len(tokenizer.word_index) + 1

# define BiLSTM model function with EarlyStopping and Dropout
def bilstm():
    model = Sequential()
    model.add(Embedding(input_dim=vocab_size, output_dim=100, input_length=max_len))
    model.add(Bidirectional(LSTM(96, dropout=0.5, recurrent_dropout=0.2, kernel_regularizer='l2')))
    model.add(Dense(96, activation='relu'))
    model.add(Dropout(0.5)) 
    model.add(Dense(len(labels), activation='softmax'))
    model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    
    # Define EarlyStopping callback
    early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
    
    # Train the model with EarlyStopping callback
    model.fit(X_train, y_train, epochs=32, batch_size=32, validation_data=(X_val, y_val), callbacks=[early_stopping])
    
    return model

# Create and train the BiLSTM model
bilstm_model = bilstm()

# Evaluate BiLSTM model on validation data
loss, accuracy = bilstm_model.evaluate(X_val, y_val)
print(f"Validation Accuracy: {accuracy:.4f}")

# Predict labels for validation data
y_val_pred = np.argmax(bilstm_model.predict(X_val), axis=1)

# Generate and print classification report
classification_rep = classification_report(y_val, y_val_pred, target_names=labels)
print("Classification Report:")
print(classification_rep)

# Function to generate bot response using BiLSTM model
def generate_bot_response(user_input, labels, threshold=0.5):
    user_input_seq = tokenizer.texts_to_sequences([user_input])
    user_input_padded = pad_sequences(user_input_seq, maxlen=max_len)
    predicted_probs = bilstm_model.predict(user_input_padded)
    predicted_label_idx = np.argmax(predicted_probs)
    max_probability = np.max(predicted_probs)
    
    if max_probability < threshold:
        response = "Sorry, I didn't understand that."
        predicted_label = None
    else:
        predicted_label = labels[predicted_label_idx]
        responses = [intent['responses'] for intent in intents['intents'] if intent['tag'] == predicted_label]
        if responses:
            response = random.choice(responses[0])
        else:
            response = "I'm unable to respond right now."
    
    return {'bot_response': response, 'predicted_label': predicted_label}

# Routes
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
        response = generate_bot_response(user_input, labels, threshold=0.8)
        
        # Print the predicted classification label
        predicted_label = response['predicted_label']
        if predicted_label:
            print(f"Classification for '{user_input}': {predicted_label}")
        
        return jsonify(response)

    except Exception as e:
        error_message = f"An error occurred: {str(e)}"
        logging.error(error_message)  # Log error to app.log
        return jsonify({'bot_response': 'Internal Server Error'}), 500

if __name__ == '__main__':
    app.run(debug=True)
