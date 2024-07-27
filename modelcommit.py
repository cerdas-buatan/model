import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
import tensorflow as tf
from transformers import DistilBertTokenizer, TFDistilBertModel
import joblib
import os

# Define folder to save model and other files
save_dir = 'saved_model'
os.makedirs(save_dir, exist_ok=True)

# Initialize list to store cleaned rows
rows = []

# Read and clean dataset
with open('data.csv', 'r', encoding='utf-8') as file:
    for line in file:
        parts = line.strip().split('|')
        if len(parts) == 2:
            rows.append(parts)

df = pd.DataFrame(rows, columns=['question', 'answer'])
df.dropna(inplace=True)

# Initialize CountVectorizer
vectorizer = CountVectorizer()
X_bow = vectorizer.fit_transform(df['question'])

# Initialize and fit LabelEncoder
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(df['answer'])

# Load tokenizer and model
tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
distilbert = TFDistilBertModel.from_pretrained('distilbert-base-uncased')

# Function to get BERT embeddings
def get_bert_embeddings(texts):
    inputs = tokenizer(texts, return_tensors='tf', padding=True, truncation=True, max_length=128)
    outputs = distilbert(**inputs)
    return outputs.last_hidden_state[:, 0, :].numpy()  # Use CLS token embedding

# Get BERT embeddings
X_bert_embeddings = get_bert_embeddings(df['question'].tolist())

# Combine BERT embeddings with BoW features
X_combined = np.concatenate([X_bert_embeddings, X_bow.toarray()], axis=1)

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X_combined, y, test_size=0.2, random_state=42)

# Create TensorFlow datasets
train_dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train)).batch(64).cache().prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
test_dataset = tf.data.Dataset.from_tensor_slices((X_test, y_test)).batch(64).cache().prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

# Set TensorFlow threading configurations
tf.config.threading.set_inter_op_parallelism_threads(4)
tf.config.threading.set_intra_op_parallelism_threads(4)

# Define and compile model
model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(X_combined.shape[1],)),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Dense(len(label_encoder.classes_), activation='softmax')
])

# Mixed Precision
tf.keras.mixed_precision.set_global_policy('mixed_float16')

# Learning rate scheduling
lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate=1e-4,
    decay_steps=1000,
    decay_rate=0.9
)

model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=lr_schedule),
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Train model
model.fit(train_dataset, epochs=10, validation_data=test_dataset)

# Save model, vectorizer, and label encoder
model.save(os.path.join(save_dir, 'combined_model.keras'))
joblib.dump(vectorizer, os.path.join(save_dir, 'vectorizer.pkl'))
joblib.dump(label_encoder, os.path.join(save_dir, 'label_encoder.pkl'))

print(f"Training complete. Model, vectorizer, and label encoder saved in '{save_dir}'.")
