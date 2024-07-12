import tensorflow as tf
from transformers import AutoTokenizer, TFAutoModelForSequenceClassification
import pandas as pd
import numpy as np  # Tambahkan impor numpy
import csv
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# Load the dataset
def filter_valid_rows(row):
    return len(row) == 2 and row[0].strip() != "" and row[1].strip() != ""

with open('./dataset/dataset_clean2.csv', 'r', encoding='utf-8') as file:
    reader = csv.reader(file, delimiter='|')
    filtered_rows = [row for row in reader if filter_valid_rows(row)]

df = pd.DataFrame(filtered_rows, columns=['question', 'answer'])
print(f"Number of valid rows: {len(df)}")

# Encode the labels (answers) to numeric values
label_encoder = LabelEncoder()
df['encoded_answer'] = label_encoder.fit_transform(df['answer'])

# Prepare the dataset
tokenizer = AutoTokenizer.from_pretrained('indolem/indobert-base-uncased')
input_ids = []
attention_masks = []

for question in df['question']:
    encoded = tokenizer.encode_plus(
        question,
        add_special_tokens=True,
        max_length=128,
        padding='max_length',
        truncation=True,
        return_attention_mask=True,
        return_tensors='tf'
    )
    input_ids.append(encoded['input_ids'].numpy())
    attention_masks.append(encoded['attention_mask'].numpy())

input_ids = np.concatenate(input_ids, axis=0)
attention_masks = np.concatenate(attention_masks, axis=0)
labels = df['encoded_answer'].values

# Split the dataset into training, validation, and test sets
train_inputs, temp_inputs, train_masks, temp_masks, train_labels, temp_labels = train_test_split(
    input_ids, attention_masks, labels, test_size=0.3, random_state=42
)

val_inputs, test_inputs, val_masks, test_masks, val_labels, test_labels = train_test_split(
    temp_inputs, temp_masks, temp_labels, test_size=0.5, random_state=42
)

# Convert to tensors
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
batch_size = 5
train_dataset = tf.data.Dataset.from_tensor_slices(((train_inputs, train_masks), train_labels)).shuffle(len(train_labels)).batch(batch_size)
val_dataset = tf.data.Dataset.from_tensor_slices(((val_inputs, val_masks), val_labels)).batch(batch_size)
test_dataset = tf.data.Dataset.from_tensor_slices(((test_inputs, test_masks), test_labels)).batch(batch_size)

# Load IndoBERT model from PyTorch weights
model = TFAutoModelForSequenceClassification.from_pretrained('indolem/indobert-base-uncased', from_pt=True, num_labels=len(label_encoder.classes_))

# Custom train step
@tf.function
def train_step(model, optimizer, loss_fn, accuracy_metric, x, y):
    with tf.GradientTape() as tape:
        input_ids, attention_masks = x
        logits = model(input_ids, attention_mask=attention_masks, training=True).logits
        loss = loss_fn(y, logits)
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    
    preds = tf.argmax(logits, axis=1, output_type=tf.int32)
    accuracy_metric.update_state(y, preds)
    
    return loss

# Compile the model
optimizer = tf.keras.optimizers.Adam(learning_rate=2e-5, epsilon=1e-8)
loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
accuracy_metric = tf.keras.metrics.Accuracy()

# Training loop with validation
epochs = 400
for epoch in range(epochs):
    print(f"Epoch {epoch + 1}/{epochs}")
    accuracy_metric.reset_states()
    # Training
    epoch_loss = 0
    for step, ((x_batch_train, x_batch_mask), y_batch_train) in enumerate(train_dataset):
        loss = train_step(model, optimizer, loss_fn, accuracy_metric, (x_batch_train, x_batch_mask), y_batch_train)
        epoch_loss += loss
        if step % 10 == 0:
            print(f"Training loss (for one batch) at step {step}: {loss:.4f}")
    train_accuracy = accuracy_metric.result()
    epoch_loss /= (step + 1)
    print(f"Training loss after epoch {epoch + 1}: {epoch_loss:.4f}")
    print(f"Training accuracy after epoch {epoch + 1}: {train_accuracy:.4f}")
    
    # Validation
    val_loss = 0
    accuracy_metric.reset_states()
    for (x_batch_val, x_mask_val), y_batch_val in val_dataset:
        logits = model(x_batch_val, attention_mask=x_mask_val, training=False).logits
        val_loss += loss_fn(y_batch_val, logits).numpy()
        
        preds = tf.argmax(logits, axis=1, output_type=tf.int32)
        accuracy_metric.update_state(y_batch_val, preds)
    
    val_loss /= len(val_dataset)
    val_accuracy = accuracy_metric.result()
    print(f"Validation loss after epoch {epoch + 1}: {val_loss:.4f}")
    print(f"Validation accuracy after epoch {epoch + 1}: {val_accuracy:.4f}")

# Save the model
model.save_pretrained('indobert_model')
tokenizer.save_pretrained('indobert_model')
