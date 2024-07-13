import tensorflow as tf
from transformers import AutoTokenizer, TFAutoModelForSequenceClassification
import pandas as pd
import numpy as np
import csv
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# Load the dataset
def filter_valid_rows(row):
    return len(row) == 2 and row[0].strip() != "" and row[1].strip() != ""

try:
    with open('./dataset/dataset_clean2.csv', 'r', encoding='utf-8') as file:
        reader = csv.reader(file, delimiter='|')
        filtered_rows = [row for row in reader if filter_valid_rows(row)]
except FileNotFoundError as e:
    print(f"Error: {e}")
    exit()

df = pd.DataFrame(filtered_rows, columns=['question', 'answer'])
print(f"Number of valid rows: {len(df)}")

# Encode the labels (answers) to numeric values
label_encoder = LabelEncoder()
df['encoded_answer'] = label_encoder.fit_transform(df['answer'])

# Prepare the dataset
tokenizer = AutoTokenizer.from_pretrained('indolem/indobert-base-uncased')
inputs = tokenizer(
    df['question'].tolist(),
    add_special_tokens=True,
    max_length=128,
    padding='max_length',
    truncation=True,
    return_attention_mask=True,
    return_tensors='np'  # Note: Changed to numpy tensors
)

input_ids = inputs['input_ids']
attention_masks = inputs['attention_mask']
labels = df['encoded_answer'].values

# Split the dataset into training, validation, and test sets
train_inputs, temp_inputs, train_masks, temp_masks, train_labels, temp_labels = train_test_split(
    input_ids, attention_masks, labels, test_size=0.3, random_state=42
)

val_inputs, test_inputs, val_masks, test_masks, val_labels, test_labels = train_test_split(
    temp_inputs, temp_masks, temp_labels, test_size=0.5, random_state=42
)

# Convert back to tensors
train_inputs = tf.constant(train_inputs)
val_inputs = tf.constant(val_inputs)
test_inputs = tf.constant(test_inputs)
train_masks = tf.constant(train_masks)
val_masks = tf.constant(val_masks)
test_masks = tf.constant(test_masks)
train_labels = tf.constant(train_labels)
val_labels = tf.constant(val_labels)
test_labels = tf.constant(test_labels)

# Convert to tf.data.Dataset
batch_size = 10
train_dataset = tf.data.Dataset.from_tensor_slices(((train_inputs, train_masks), train_labels)).shuffle(len(train_labels)).batch(batch_size)
val_dataset = tf.data.Dataset.from_tensor_slices(((val_inputs, val_masks), val_labels)).batch(batch_size)
test_dataset = tf.data.Dataset.from_tensor_slices(((test_inputs, test_masks), test_labels)).batch(batch_size)

# Load IndoBERT model from PyTorch weights
model = TFAutoModelForSequenceClassification.from_pretrained('indolem/indobert-base-uncased', from_pt=True, num_labels=len(label_encoder.classes_))

# Compile the model
optimizer = tf.keras.optimizers.Adam(learning_rate=2e-5, epsilon=1e-8)
loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
accuracy_metric = tf.keras.metrics.SparseCategoricalAccuracy('accuracy')

model.compile(optimizer=optimizer, loss=loss_fn, metrics=[accuracy_metric])

# Callbacks
callbacks = [
    tf.keras.callbacks.ModelCheckpoint(filepath='best_model.h5', save_best_only=True, monitor='val_loss', mode='min'),
    tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
]

# Train the model
history = model.fit(
    train_dataset,
    epochs=500,
    validation_data=val_dataset,
    callbacks=callbacks
)

# Save the model
model.save_pretrained('indobert_model')
tokenizer.save_pretrained('indobert_model')

# Evaluate the model on the test set
test_loss, test_accuracy = model.evaluate(test_dataset)
print(f"Test Loss: {test_loss:.4f}")
print(f"Test Accuracy: {test_accuracy:.4f}")

# delsoon
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

# Define a custom accuracy function
def masked_accuracy(y_true, y_pred):
    y_true = tf.cast(tf.reshape(y_true, (-1,)), tf.int64)
    y_pred = tf.cast(tf.argmax(y_pred, axis=-1), tf.int64)
    y_pred = tf.reshape(y_pred, (-1,))  # Ensure y_pred is reshaped to match y_true
    accuracy = tf.equal(y_true, y_pred)
    mask = tf.cast(tf.not_equal(y_true, tokenizer.pad_token_id), tf.float32)  # Ignore padding tokens
    accuracy = tf.cast(accuracy, tf.float32) * mask
    return tf.reduce_sum(accuracy) / tf.reduce_sum(mask)

