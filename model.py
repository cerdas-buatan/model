# import tensorflow as tf
# from transformers import TFBertForSequenceClassification, BertTokenizer
# import pandas as pd

# # Load the dataset with proper delimiter
# df = pd.read_csv('dataset-a-NewBackup.csv', delimiter='|', names=['question|answer'])

# # Split the question and answer into separate columns
# df[['question', 'answer']] = df['question|answer'].str.split('|', expand=True)

# # Prepare the dataset
# tokenizer = BertTokenizer.from_pretrained('indobenchmark/indobert-base-p2')
# input_ids = []
# attention_masks = []

# for question in df['question']:
#     encoded = tokenizer.encode_plus(question, add_special_tokens=True, max_length=64, padding='max_length', return_attention_mask=True, truncation=True)
#     input_ids.append(encoded['input_ids'])
#     attention_masks.append(encoded['attention_mask'])

# input_ids = tf.constant(input_ids)
# attention_masks = tf.constant(attention_masks)
# labels = tf.constant(df['answer'].values)

# # Load model
# model = TFBertForSequenceClassification.from_pretrained('indobenchmark/indobert-base-p2', num_labels=len(df['answer'].unique()))

# # Compile and train the model
# model.compile(optimizer=tf.keras.optimizers.Adam(), loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), metrics=['accuracy'])
# model.fit([input_ids, attention_masks], labels, epochs=3, batch_size=32)

# # Save the model
# model.save_pretrained('indobert_model')
# tokenizer.save_pretrained('indobert_model')



import pandas as pd
import re
from transformers import TFBertForSequenceClassification, BertTokenizer
import tensorflow as tf
from sklearn.model_selection import train_test_split

# Load the dataset
df = pd.read_csv('dataset-a-NewBackup.csv')

# Clean and combine the dataset
def clean_text(text):
    text = re.sub(r'[^\w\s]', '', text)  # Remove punctuation
    text = re.sub(r'\s+', ' ', text).strip()  # Remove extra spaces
    return text

df['question'] = df['question'].apply(clean_text)
df['answer'] = df['answer'].apply(clean_text)
df['question|answer'] = df['question'] + '|' + df['answer']

# Prepare the dataset
tokenizer = BertTokenizer.from_pretrained('indobenchmark/indobert-base-p2')
input_ids = []
attention_masks = []
labels = []

for idx, row in df.iterrows():
    question, answer = row['question'], row['answer']
    encoded = tokenizer.encode_plus(
        question, add_special_tokens=True, max_length=64, 
        padding='max_length', return_attention_mask=True, truncation=True
    )
    input_ids.append(encoded['input_ids'])
    attention_masks.append(encoded['attention_mask'])
    labels.append(answer)  # Assuming answers are already in a numeric format

input_ids = tf.constant(input_ids)
attention_masks = tf.constant(attention_masks)
labels = tf.constant(labels)

# Split the dataset into training and testing sets
train_input_ids, test_input_ids, train_attention_masks, test_attention_masks, train_labels, test_labels = train_test_split(
    input_ids, attention_masks, labels, test_size=0.2, random_state=42)

# Load model
model = TFBertForSequenceClassification.from_pretrained('indobenchmark/indobert-base-p2', num_labels=len(set(labels.numpy())))

# Compile the model
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=2e-5), 
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), 
              metrics=['accuracy'])

# Train the model
history = model.fit(
    [train_input_ids, train_attention_masks], train_labels, 
    validation_data=([test_input_ids, test_attention_masks], test_labels), 
    epochs=3, batch_size=32
)

# Save the model
model.save_pretrained('indobert_model')
tokenizer.save_pretrained('indobert_model')

# Evaluate the model
loss, accuracy = model.evaluate([test_input_ids, test_attention_masks], test_labels)
print(f'Test Loss: {loss}')
print(f'Test Accuracy: {accuracy}')
