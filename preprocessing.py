import tensorflow as tf
from transformers import TFBertForSequenceClassification, BertTokenizer
from sklearn.model_selection import train_test_split
import pandas as pd

# Load the dataset
df = pd.read_csv('./dataset/a-backup.csv')

# Prepare the dataset
tokenizer = BertTokenizer.from_pretrained('indobenchmark/indobert-base-p2')
input_ids = []
attention_masks = []

for question in df['question']:
    encoded = tokenizer.encode_plus(question, add_special_tokens=True, max_length=64, pad_to_max_length=True, return_attention_mask=True)
    input_ids.append(encoded['input_ids'])
    attention_masks.append(encoded['attention_mask'])

input_ids = tf.constant(input_ids)
attention_masks = tf.constant(attention_masks)
labels = tf.constant(df['answer'].values)

# Split the dataset into training and testing sets
train_input_ids, test_input_ids, train_attention_masks, test_attention_masks, train_labels, test_labels = train_test_split(
    input_ids, attention_masks, labels, test_size=0.2, random_state=42)

# Load model
model = TFBertForSequenceClassification.from_pretrained('indobenchmark/indobert-base-p2', num_labels=len(df['answer'].unique()))

# Compile the model
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=2e-5), 
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), 
              metrics=['accuracy'])

# Train the model
history = model.fit([train_input_ids, train_attention_masks], train_labels, 
                    validation_data=([test_input_ids, test_attention_masks], test_labels), 
                    epochs=3, batch_size=32)

# Save the model
model.save_pretrained('indobert_model')
tokenizer.save_pretrained('indobert_model')
