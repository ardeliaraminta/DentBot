import numpy as np
import tensorflow as tf
from pymongo import MongoClient
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Bidirectional, LSTM, Dense
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import EarlyStopping
import os
import matplotlib.pyplot as plt

# Connect to MongoDB
uri = "mongodb+srv://Visitor:researchgogogo@reseearch.a2rwr6l.mongodb.net/"
client = MongoClient(uri)
db = client["research"]
intents_collection = db["data"]

# Load intents data from MongoDB
intents_data = list(intents_collection.find())
labels = list(set(intent['tag'] for intent in intents_data))

X = []
y = []
for intent in intents_data:
    for pattern in intent['patterns']:
        X.append(pattern)
        y.append(intent['tag'])

tokenizer = Tokenizer()
tokenizer.fit_on_texts(X)
X_sequences = tokenizer.texts_to_sequences(X)
max_len = max(len(seq) for seq in X_sequences)
X_padded = pad_sequences(X_sequences, maxlen=max_len)
y_encoded = np.array([labels.index(label) for label in y])


def bi_lstm(vocab_size, embedding_dim=200, lstm_units=96, dropout_rate=0.8):
    model = Sequential()
    model.add(Embedding(input_dim=vocab_size, output_dim=embedding_dim))
    model.add(Bidirectional(LSTM(lstm_units, dropout=dropout_rate, recurrent_dropout=dropout_rate)))
    model.add(Dense(len(labels), activation='softmax'))
    model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

dropout_rate = 0.2  # experiment with different dropout rates (e.g., 0.2, 0.3, 0.5)
# train-test and get vocabulary size using tokenizer
X_train, X_val, y_train, y_val = train_test_split(X_padded, y_encoded, test_size=0.2, random_state=42)
vocab_size = len(tokenizer.word_index) + 1  # +1 for OOV token
bi_lstm_model = bi_lstm(vocab_size, dropout_rate=dropout_rate)

early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
history = bi_lstm_model.fit(X_train, y_train, epochs=32, batch_size=32, 
                             validation_data=(X_val, y_val), callbacks=[early_stopping])

# Plot training and validation accuracy
train_accuracy = history.history['accuracy']
val_accuracy = history.history['val_accuracy']
epochs = range(1, len(train_accuracy) + 1)

plt.figure(figsize=(10, 6))
plt.plot(epochs, train_accuracy, label='Training Accuracy', color='blue', linestyle='-', linewidth=2)
plt.plot(epochs, val_accuracy, label='Validation Accuracy', color='orange', linestyle='--', linewidth=2)

plt.xlabel('Epoch', fontsize=12)
plt.ylabel('Accuracy', fontsize=12)
plt.title('Bi-LSTM Model Training and Validation Accuracy', fontsize=14)
plt.ylim(0, 1)
plt.legend(loc='lower right', fontsize=10)
plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()
plt.show()
