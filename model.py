import tensorflow as tf
from transformers import TFBertForSequenceClassification, BertTokenizer

# Load the dataset
# Assuming the dataset is in a CSV format with 'question' and 'answer' columns
import pandas as pd
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

# Load model
model = TFBertForSequenceClassification.from_pretrained('indobenchmark/indobert-base-p2')

# Compile and train the model
model.compile(optimizer=tf.keras.optimizers.Adam(), loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), metrics=['accuracy'])
model.fit([input_ids, attention_masks], labels, epochs=3, batch_size=32)

# Save the model
model.save('indobert_model')