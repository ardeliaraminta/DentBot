import numpy as np
from pymongo import MongoClient
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from tensorflow.keras.callbacks import EarlyStopping

# Connect to MongoDB
uri = "mongodb+srv://Visitor:researchgogogo@reseearch.a2rwr6l.mongodb.net/"
client = MongoClient(uri)
db = client["research"]
intents_collection = db["data"]
intents_data = list(intents_collection.find())
labels = list(set(intent['tag'] for intent in intents_data))

# Extract patterns and tags from intents data
X = []
y = []
for intent in intents_data:
    for pattern in intent['patterns']:
        X.append(pattern)
        y.append(intent['tag'])

# Tokenization and padding for LSTM input
tokenizer = Tokenizer()
tokenizer.fit_on_texts(X)
X_sequences = tokenizer.texts_to_sequences(X)
max_len = max(len(seq) for seq in X_sequences)
X_padded = pad_sequences(X_sequences, maxlen=max_len)
y_encoded = np.array([labels.index(label) for label in y])


def lstm(vocab_size, embedding_dim=100, lstm_units=128):
    model = Sequential()
    model.add(Embedding(input_dim=vocab_size, output_dim=embedding_dim))
    model.add(LSTM(lstm_units))
    model.add(Dense(len(labels), activation='softmax'))
    model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

X_train, X_val, y_train, y_val = train_test_split(X_padded, y_encoded, test_size=0.2, random_state=42)

vocab_size = len(tokenizer.word_index) + 1  # +1 for OOV token
lstm_model = lstm(vocab_size)
early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

# Train the model with early stopping
lstm_model.fit(X_train, y_train, epochs=40, batch_size=32, validation_data=(X_val, y_val), callbacks=[early_stopping])

# Evaluate the model
y_pred = np.argmax(lstm_model.predict(X_padded), axis=1)
print(classification_report(y_encoded, y_pred, target_names=labels))

