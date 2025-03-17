import json
import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Bidirectional, LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
import os

data_path = os.path.join(os.path.dirname(__file__), 'new.json')

# Load intents data from JSON file
with open(data_path, 'r') as json_data:
    intents = json.load(json_data)

# Extract patterns and labels from intents data
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

# Convert labels to numeric format
labels = list(set(y))
label_to_idx = {label: idx for idx, label in enumerate(labels)}
y_numeric = np.array([label_to_idx[label] for label in y])

# Split data into train and validation sets
X_train, X_val, y_train, y_val = train_test_split(X_padded, y_numeric, test_size=0.2, random_state=42)

vocab_size = len(tokenizer.word_index) + 1
embedding_dim = 100

# Define the BiLSTM model
model = Sequential()
model.add(Embedding(input_dim=vocab_size, output_dim=embedding_dim, input_length=max_len))
model.add(Bidirectional(LSTM(32, dropout=0.5, recurrent_dropout=0.2, kernel_regularizer='l2')))
model.add(Dense(32, activation='relu', kernel_regularizer='l2'))
model.add(Dropout(0.5))
model.add(Dense(len(labels), activation='softmax'))

# Compile model
model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Define EarlyStopping callback
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

# Train model with EarlyStopping callback
history = model.fit(X_train, y_train, epochs=32, batch_size=8, validation_data=(X_val, y_val), callbacks=[early_stopping])

# Evaluate the model on validation data and generate predictions
y_val_probs = model.predict(X_val)
y_val_pred_labels = [labels[np.argmax(prob)] for prob in y_val_probs]
y_val_true_labels = [labels[idx] for idx in y_val]

# Print classification report
report = classification_report(y_val_true_labels, y_val_pred_labels)
print("Classification Report:")
print(report)

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

# Calculate confusion matrix
conf_matrix = confusion_matrix(y_val_true_labels, y_val_pred_labels, labels=labels)

# Plot confusion matrix
plt.figure(figsize=(12, 10))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.title('Confusion Matrix')
plt.show()
