import json
import os
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
import numpy as np

# Load intents from data.json
data_path = os.path.join(os.path.dirname(__file__), 'data.json')
with open(data_path, 'r') as file:
    intents = json.load(file)

# Extract patterns and tags from intents data for training
X = []
y = []
for intent in intents['intents']:
    for pattern in intent['patterns']:
        X.append(pattern)
        y.append(intent['tag'])

# Vectorize text data using TF-IDF
vectorizer = TfidfVectorizer()
X_tfidf = vectorizer.fit_transform(X)

# Encode labels
labels = list(set(y))
label_to_idx = {label: idx for idx, label in enumerate(labels)}
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

# Save the trained model and vectorizer using pickle
model_path = os.path.join(os.path.dirname(__file__), 'model.pkl')
vectorizer_path = os.path.join(os.path.dirname(__file__), 'vectorizer.pkl')
with open(model_path, 'wb') as file:
    pickle.dump(best_svm_model, file)
with open(vectorizer_path, 'wb') as file:
    pickle.dump(vectorizer, file)
