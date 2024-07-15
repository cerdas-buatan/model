import json
import os
import pandas as pd
import tensorflow as tf
import requests
from transformers import TFAutoModelForSeq2SeqLM, AutoTokenizer
from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard
from tensorflow.keras.preprocessing.sequence import pad_sequences

## Path to the dataset file
dataset_path = 'dataset_clean2.csv'

# URL of the dataset in the GitHub repository
url = 'https://raw.githubusercontent.com/cerdas-buatan/dataset/main/dataset_clean2.csv'

# Check if the dataset already exists locally
if not os.path.exists(dataset_path):
    # Download the dataset
    response = requests.get(url)
    if response.status_code == 200:
        # Save the dataset to a local file
        with open(dataset_path, 'wb') as file:
            file.write(response.content)
    else:
        raise Exception(f"Failed to download dataset: {response.status_code}")

# Initialize an empty list to store cleaned rows
rows = []

# Read and clean the dataset, handling any anomalies
with open(dataset_path, 'r', encoding='utf-8') as file:
    for line_number, line in enumerate(file):
        # Split line by '|' and handle any unexpected lines
        parts = line.strip().split('|')
        if len(parts) == 2:  # Only process lines with exactly two parts
            rows.append(parts)

# Convert cleaned rows to a DataFrame
df = pd.DataFrame(rows, columns=['question', 'answer'])

# Ensure the dataset has at least two columns
if df.shape[1] < 2:
    raise ValueError("The dataset does not have the expected number of columns.")

# Split the dataset into questions and answers
questions_train = df.iloc[:1794, 0].values.tolist()
answers_train = df.iloc[:1794, 1].values.tolist()

questions_test = df.iloc[1794:, 0].values.tolist()
answers_test = df.iloc[1794:, 1].values.tolist()

def ensure_string(texts):
    return [str(text) for text in texts]

questions_train = ensure_string(questions_train)
answers_train = ensure_string(answers_train)
questions_test = ensure_string(questions_test)
answers_test = ensure_string(answers_test)

# Initialize the tokenizer
tokenizer = AutoTokenizer.from_pretrained('indobert/indobertseq2seq')

# Tokenize the input and output sequences
def tokenize_sequences(tokenizer, texts, max_length=64):
    return tokenizer(
        texts,
        max_length=max_length,
        padding='max_length',
        truncation=True,
        return_tensors='tf'
    )

train_encodings = tokenize_sequences(tokenizer, questions_train)
train_decodings = tokenize_sequences(tokenizer, answers_train)
test_encodings = tokenize_sequences(tokenizer, questions_test)
test_decodings = tokenize_sequences(tokenizer, answers_test)

# Load the sequence-to-sequence model
model = TFAutoModelForSeq2SeqLM.from_pretrained('indobert/indobertseq2seq')

# Define the optimizer
optimizer = tf.keras.optimizers.Adam(learning_rate=5e-5)

# Define a custom accuracy function
def masked_accuracy(y_true, y_pred):
    y_true = tf.cast(tf.reshape(y_true, (-1,)), tf.int64)
    y_pred = tf.cast(tf.argmax(y_pred, axis=-1), tf.int64)
    y_pred = tf.reshape(y_pred, (-1,))  # Ensure y_pred is reshaped to match y_true
    accuracy = tf.equal(y_true, y_pred)
    mask = tf.cast(tf.not_equal(y_true, tokenizer.pad_token_id), tf.float32)  # Ignore padding tokens
    accuracy = tf.cast(accuracy, tf.float32) * mask
    return tf.reduce_sum(accuracy) / tf.reduce_sum(mask)

# Compile the model with the custom accuracy metric
model.compile(optimizer=optimizer, loss=model.compute_loss, metrics=[masked_accuracy])

# Create a tf.data.Dataset
train_dataset = tf.data.Dataset.from_tensor_slices((
    {
        'input_ids': train_encodings['input_ids'],
        'attention_mask': train_encodings['attention_mask'],
        'decoder_input_ids': train_decodings['input_ids'],
        'decoder_attention_mask': train_decodings['attention_mask']
    },
    train_decodings['input_ids']
)).batch(64)

test_dataset = tf.data.Dataset.from_tensor_slices((
    {
        'input_ids': test_encodings['input_ids'],
        'attention_mask': test_encodings['attention_mask'],
        'decoder_input_ids': test_decodings['input_ids'],
        'decoder_attention_mask': test_decodings['attention_mask']
    },
    test_decodings['input_ids']
)).batch(64)

# Setup the callbacks
path = "output_dir/"
os.makedirs(path, exist_ok=True)

logdir = os.path.join(path, "logs")
tensorboard_callback = TensorBoard(logdir, histogram_freq=1)

checkpoint = ModelCheckpoint(
    os.path.join(path, 'model-{epoch:02d}-{loss:.2f}.hdf5'),
    monitor='loss',
    verbose=1,
    save_best_only=True,
    mode='auto',
    save_freq=150
)

# Train the model for more epochs
model.fit(train_dataset, epochs=100, validation_data=test_dataset, callbacks=[tensorboard_callback, checkpoint])

# Save the model and tokenizer
model_path = 'seqinbert_model'
model.save_pretrained(model_path)
tokenizer.save_pretrained(model_path)

# Load the model and tokenizer for generating text
model = TFAutoModelForSeq2SeqLM.from_pretrained(model_path)
tokenizer = AutoTokenizer.from_pretrained(model_path)

# # Define a function to generate text based on input
# def generate_text(input_text):
#     input_ids = tokenizer.encode(input_text, return_tensors='tf', max_length=64, truncation=True, padding='max_length')
#     output_ids = model.generate(input_ids=input_ids, max_length=64, num_beams=5, early_stopping=True)
#     output_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
#     return output_text

# # Example usage
# while True:
#     input_text = input("Masukkan pertanyaan Anda (atau ketik 'exit' untuk keluar): ")

#     if input_text.lower() == 'exit':
#         break

#     # Generate text based on input
#     generated_text = generate_text(input_text)
#     print("Jawaban dari model:")
