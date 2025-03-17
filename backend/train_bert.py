from transformers import BertTokenizer, BertForSequenceClassification, TrainingArguments, Trainer
from sklearn.model_selection import train_test_split
from pymongo import MongoClient

uri = "mongodb+srv://Visitor:researchgogogo@reseearch.a2rwr6l.mongodb.net/"
client = MongoClient(uri)
db = client["research"]
intents_collection = db["data"]
intents_data = list(intents_collection.find())
labels = list(set(intent['tag'] for intent in intents_data))

X = []
y = []
# iterate over intents data to collect patterns and corresponding tags
for intent in intents_data:
    for pattern in intent['patterns']:
        X.append(pattern)
        y.append(intent['tag'])

# tokenize the data using bert tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
X_encodings = tokenizer(X, padding=True, truncation=True, return_tensors='pt')

# convert labels to numerical indices
label_to_index = {label: i for i, label in enumerate(labels)}
y_encoded = [label_to_index[label] for label in y]

# split data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X_encodings['input_ids'], y_encoded, test_size=0.2, random_state=42)

# define training arguments
training_args = TrainingArguments(
    output_dir='./results',
    evaluation_strategy='epoch',
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=3,
    logging_dir='./logs',
    logging_steps=100,
)


bert = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=len(labels))

# create trainer
trainer = Trainer(
    model=bert,
    args=training_args,
    train_dataset=list(zip(X_train, y_train)),
    eval_dataset=list(zip(X_val, y_val)),
)

# perform hyperparameter search (example: learning_rate)
best_run = trainer.hyperparameter_search(
    direction='maximize', 
    hp_space={'learning_rate': (1e-5, 5e-5, 'log-uniform')},
)

print(best_run)
