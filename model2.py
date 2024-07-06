import tensorflow as tf
from transformers import TFT5ForConditionalGeneration, T5Tokenizer
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

# Initialize the tokenizer
tokenizer = T5Tokenizer.from_pretrained('t5-small')

# Tokenize the input and output texts
input_ids = []
attention_masks = []
labels = []

for question, answer in zip(df['question'], df['answer']):
    input_encoded = tokenizer.encode_plus(question, add_special_tokens=True, max_length=64, padding='max_length', return_attention_mask=True, truncation=True)
    output_encoded = tokenizer.encode_plus(answer, add_special_tokens=True, max_length=64, padding='max_length', return_attention_mask=True, truncation=True)
    
    input_ids.append(input_encoded['input_ids'])
    attention_masks.append(input_encoded['attention_mask'])
    labels.append(output_encoded['input_ids'])

# Convert lists to tensors
input_ids = tf.constant(input_ids)
attention_masks = tf.constant(attention_masks)
labels = tf.constant(labels)

# Load the model
model = TFT5ForConditionalGeneration.from_pretrained('t5-small')

# Compile the model
optimizer = tf.keras.optimizers.Adam(learning_rate=3e-5)
model.compile(optimizer=optimizer, loss=model.compute_loss)

# Train the model
model.fit([input_ids, attention_masks], labels, epochs=3, batch_size=8)

# Save the model and tokenizer
model_path = 't5_text_to_text_model'
model.save_pretrained(model_path)
tokenizer.save_pretrained(model_path)
