import tensorflow as tf
from transformers import TFBertForSequenceClassification, BertTokenizer
import pandas as pd

# Initialize an empty list to store cleaned rows
rows = []

# Read and clean the dataset, handling any anomalies
with open('dataset-a.csv', 'r', encoding='utf-8') as file:
    for line_number, line in enumerate(file):
        # Split line by '|' and handle any unexpected lines
        parts = line.strip().split('|')
        if len(parts) == 2:  # Only process lines with exactly two parts
            rows.append(parts)

# Convert cleaned rows to a DataFrame
df = pd.DataFrame(rows, columns=['question', 'answer'])

# Convert 'answer' column to integers
df['answer'] = pd.to_numeric(df['answer'], errors='coerce').fillna(0).astype(int)

# Prepare the dataset
tokenizer = BertTokenizer.from_pretrained('indobenchmark/indobert-base-p2')
input_ids = []
attention_masks = []

for question in df['question']:
    encoded = tokenizer.encode_plus(question, add_special_tokens=True, max_length=64, padding='max_length', return_attention_mask=True, truncation=True)
    input_ids.append(encoded['input_ids'])
    attention_masks.append(encoded['attention_mask'])

input_ids = tf.constant(input_ids)
attention_masks = tf.constant(attention_masks)
labels = tf.constant(df['answer'].values)

# Load model
model = TFBertForSequenceClassification.from_pretrained('indobenchmark/indobert-base-p2', num_labels=len(df['answer'].unique()))

# Compile and train the model
model.compile(optimizer=tf.keras.optimizers.Adam(), loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), metrics=['accuracy'])
model.fit([input_ids, attention_masks], labels, epochs=3, batch_size=32)

# Save the model and tokenizer
model_path = 'indobert_model'
model.save_pretrained(model_path)
tokenizer.save_pretrained(model_path)
