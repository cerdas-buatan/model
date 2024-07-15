import tensorflow as tf
from transformers import TFT5ForConditionalGeneration, T5Tokenizer
import pandas as pd
import os
import requests

# Path to the dataset file
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

# Initialize the tokenizer
tokenizer = T5Tokenizer.from_pretrained('t5-small')

# Tokenize the input and output sequences
input_ids = []
attention_masks = []
labels = []

for index, row in df.iterrows():
    encoded_input = tokenizer.encode_plus(row['question'], add_special_tokens=True, max_length=64, padding='max_length', return_attention_mask=True, truncation=True)
    encoded_output = tokenizer.encode_plus(row['answer'], add_special_tokens=True, max_length=64, padding='max_length', return_attention_mask=True, truncation=True)

    input_ids.append(encoded_input['input_ids'])
    attention_masks.append(encoded_input['attention_mask'])
    # Shift the labels to the right for the model
    label_ids = encoded_output['input_ids']
    label_ids = [tokenizer.pad_token_id] + label_ids[:-1]  # Shift right
    labels.append(label_ids)

input_ids = tf.constant(input_ids)
attention_masks = tf.constant(attention_masks)
labels = tf.constant(labels)

# Load the sequence-to-sequence model
model = TFT5ForConditionalGeneration.from_pretrained('t5-small')

# Define the optimizer
optimizer = tf.keras.optimizers.Adam(learning_rate=5e-5)

# Define a custom loss function
def compute_loss(labels, logits):
    return tf.keras.losses.sparse_categorical_crossentropy(labels, logits, from_logits=True)

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
model.compile(optimizer=optimizer, loss=compute_loss, metrics=[masked_accuracy])

# Create a tf.data.Dataset
dataset = tf.data.Dataset.from_tensor_slices((
    {
        'input_ids': input_ids,
        'attention_mask': attention_masks,
        'labels': labels
    },
    labels
)).batch(30)

# Train the model for more epochs
model.fit(dataset, epochs=100)

# Save the model and tokenizer
model_path = 't5_text_to_text_model'
model.save_pretrained(model_path)
tokenizer.save_pretrained(model_path)

# Load the model and tokenizer for generating text
model = TFT5ForConditionalGeneration.from_pretrained(model_path)
tokenizer = T5Tokenizer.from_pretrained(model_path)

# Define a function to generate text based on input
def generate_text(input_text):
    input_ids = tokenizer.encode(input_text, return_tensors='tf', max_length=64, truncation=True, padding='max_length')
    output_ids = model.generate(input_ids=input_ids, max_length=64, num_beams=5, early_stopping=True)
    output_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    return output_text

# Example usage
while True:
    input_text = input("Masukkan pertanyaan Anda (atau ketik 'exit' untuk keluar): ")

    if input_text.lower() == 'exit':
        break

    # Generate text based on input
    generated_text = generate_text(input_text)
    print("Jawaban dari model:")
    print(generated_text)
