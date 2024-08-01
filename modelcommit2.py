import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import tensorflow as tf
import joblib
import os
import json
import re
from transformers import BertTokenizer, TFBertModel
import uuid

# Define folder to save model and other files
save_dir = 'save_model2'
os.makedirs(save_dir, exist_ok=True)

# Inisialisasi daftar kosong untuk menyimpan baris yang telah dibersihkan
rows = []

# Read and clean dataset, handling any anomalies
with open('data.csv', 'r', encoding='utf-8') as file:
    for line_number, line in enumerate(file):
        # Pisahkan baris berdasarkan '|'
        parts = line.strip().split('|')
        if len(parts) == 2:  # Hanya memproses baris dengan tepat dua bagian
            rows.append(parts)

# Konversi baris yang telah dibersihkan ke DataFrame
df = pd.DataFrame(rows, columns=['question', 'answer'])

# Mengatasi missing values
df.dropna(inplace=True)

# Remove duplicates
df.drop_duplicates(inplace=True)

# Text preprocessing function
def preprocess_text(text):
    # Convert text to lowercase
    text = text.lower()
    # Remove punctuation and numbers
    text = re.sub(r'[^a-z\s]', '', text)
    # Strip leading and trailing whitespace
    text = text.strip()
    return text

# Apply text preprocessing
df['question'] = df['question'].apply(preprocess_text)
df['answer'] = df['answer'].apply(preprocess_text)

# Load IndoBERT model and tokenizer
tokenizer = BertTokenizer.from_pretrained('indobenchmark/indobert-base-p1')
indobert = TFBertModel.from_pretrained('indobenchmark/indobert-base-p1')

# Tokenize the text data
def tokenize_text(texts, max_length=128):
    return tokenizer(texts.tolist(), padding='max_length', truncation=True, max_length=max_length, return_tensors='tf')

tokens = tokenize_text(df['question'])

# Extract embeddings from IndoBERT
embeddings = indobert(tokens['input_ids'], attention_mask=tokens['attention_mask'])[0]

# Use the CLS token embeddings (first token) for classification
X = embeddings[:, 0, :]

# Encode labels
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(df['answer'])

# Split data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define Neural Network model
model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(X_train.shape[1],)),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(len(label_encoder.classes_), activation='softmax')
])

# Compile model
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Train model
model.fit(X_train, y_train, epochs=10, batch_size=32, validation_split=0.1)

# Save model, tokenizer, and label encoder
model.save(os.path.join(save_dir, 'nn_model.h5'))
joblib.dump(tokenizer, os.path.join(save_dir, 'tokenizer.pkl'))
joblib.dump(label_encoder, os.path.join(save_dir, 'label_encoder.pkl'))

print(f"Training complete. Model, tokenizer, and label encoder saved in '{save_dir}'.")

# Predict on test data
predictions = model.predict(X_test)
predicted_labels = label_encoder.inverse_transform(tf.argmax(predictions, axis=1).numpy())

# Create JSON output for MongoDB
output = []
for question, answer in zip(df['question'].iloc[X_test.indices], predicted_labels):
    output.append({
        "_id": {"$oid": str(uuid.uuid4())},
        "message": question + " | " + answer
    })

# Save output to JSON file
with open('gaysdisal.json', 'w', encoding='utf-8') as f:
    json.dump(output, f, ensure_ascii=False, indent=4)

print(f"Output JSON saved to 'gaysdisal.json'")
