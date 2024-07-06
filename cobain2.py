import tensorflow as tf
from transformers import BertTokenizer, TFBertForSequenceClassification
import pandas as pd

# Load the dataset
df = pd.read_csv('dataset_clean.csv', sep='|', error_bad_lines=False, on_bad_lines='skip')

# Check the column names to ensure correct referencing
print(df.columns)

# Assuming the column names need adjustment based on the printed output
# If the actual column name is different, adjust it accordingly
df.columns = df.columns.str.strip()  # Clean up any leading or trailing spaces

# Initialize tokenizer
tokenizer = BertTokenizer.from_pretrained('indobenchmark/indobert-base-p2')

# Tokenize the input texts
input_ids = []
attention_masks = []

for question in df['question']:  # Assuming 'question' is the correct column name
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
