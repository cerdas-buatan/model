import tensorflow as tf
from transformers import TFBertForSequenceClassification, BertTokenizer
import pandas as pd

# Read and clean the dataset manually
rows = []
with open('dataset-a-NewBackup.csv', 'r', encoding='utf-8') as file:
    for line in file:
        if line.count('|') == 1: 
            rows.append(line.strip())

# Convert cleaned rows to a DataFrame
df = pd.DataFrame([row.split('|') for row in rows], columns=['question', 'answer'])

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

# Save the model
model.save_pretrained('indobert_model')
tokenizer.save_pretrained('indobert_model')
