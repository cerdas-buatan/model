import pandas as pd
import re
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, SimpleRNN, Dense
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split

# Define a function to clean text
def clean_text(text):
    if pd.isna(text):  # Handle NaN values
        return ""
    text = text.replace("iteung", "gays")
    text = re.sub(r'[^A-Za-z\s]', '', text)  # Remove non-alphabetic characters
    text = re.sub(r'\s+', ' ', text).strip()  # Remove extra spaces
    return text

# Function to prepare data for RNN
def prepare_data(df):
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(df['question'])
    tokenizer.fit_on_texts(df['answer'])
    
    vocab_size = len(tokenizer.word_index) + 1

    X = tokenizer.texts_to_sequences(df['question'])
    y = tokenizer.texts_to_sequences(df['answer'])

    max_len = max(max(len(x) for x in X), max(len(y) for y in y))

    X_pad = pad_sequences(X, maxlen=max_len, padding='post')
    y_pad = pad_sequences(y, maxlen=max_len, padding='post')

    y_pad = to_categorical(y_pad, num_classes=vocab_size)

    return X_pad, y_pad, tokenizer, vocab_size, max_len

# Build RNN model
def build_rnn_model(vocab_size, max_len):
    model = Sequential([
        Embedding(input_dim=vocab_size, output_dim=64, input_length=max_len),
        SimpleRNN(128, return_sequences=True),
        Dense(vocab_size, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

# Train the RNN model
def train_rnn_model(df):
    X_pad, y_pad, tokenizer, vocab_size, max_len = prepare_data(df)
    X_train, X_val, y_train, y_val = train_test_split(X_pad, y_pad, test_size=0.1, random_state=42)
    
    model = build_rnn_model(vocab_size, max_len)
    model.fit(X_train, y_train, epochs=5, batch_size=64, validation_data=(X_val, y_val))
    
    model.save('rnn_text_normalization_model.h5')
    return tokenizer

# Load dataset
df = pd.read_csv('data.csv', sep='|', dtype={'question': 'string', 'answer': 'string'})

# Clean the dataset
df['question'] = df['question'].apply(clean_text)
df['answer'] = df['answer'].apply(clean_text)

# Train the RNN model
tokenizer = train_rnn_model(df)

# Normalize text using the trained RNN model
def normalize_text_rnn(texts, tokenizer, model, batch_size=1024):
    sequences = tokenizer.texts_to_sequences(texts)
    max_len = max(len(seq) for seq in sequences)
    sequences_pad = pad_sequences(sequences, maxlen=max_len, padding='post')

    normalized_texts = []
    num_batches = len(sequences_pad) // batch_size + 1

    for i in range(num_batches):
        start = i * batch_size
        end = start + batch_size
        batch_sequences = sequences_pad[start:end]

        predictions = model.predict(batch_sequences)
        
        for pred in predictions:
            seq = np.argmax(pred, axis=-1)
            text = tokenizer.sequences_to_texts([seq])[0]
            normalized_texts.append(text)
    
    return normalized_texts

# Load the trained RNN model
model = tf.keras.models.load_model('rnn_text_normalization_model.h5')

# Apply normalization to the dataset
df = pd.read_csv('dataset_clean.csv', sep='|', dtype={'question': 'string', 'answer': 'string'})
df['question'] = normalize_text_rnn(df['question'].fillna(''), tokenizer, model)
df['answer'] = normalize_text_rnn(df['answer'].fillna(''), tokenizer, model)

# Save the normalized dataset
df.to_csv('dataset_normalized.csv', sep='|', index=False)
print("Data normalized and saved successfully.")
