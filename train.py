import tensorflow as tf
from transformers import AutoTokenizer, TFAutoModelForSequenceClassification
import pandas as pd
import csv
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# Load the dataset
def filter_valid_rows(row):
    return len(row) == 2 and row[0].strip() != "" and row[1].strip() != ""

with open('./dataset/dataset_sample.csv', 'r', encoding='utf-8') as file:
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
    input_ids.append(encoded['input_ids'])
    attention_masks.append(encoded['attention_mask'])

input_ids = tf.concat(input_ids, axis=0)
attention_masks = tf.concat(attention_masks, axis=0)
labels = tf.constant(df['encoded_answer'].values)

# Split the dataset into training, validation, and test sets
train_inputs_idx, temp_inputs_idx, train_masks_idx, temp_masks_idx, train_labels_idx, temp_labels_idx = train_test_split(
    range(len(input_ids)), range(len(attention_masks)), range(len(labels)), test_size=0.3, random_state=42
)

val_inputs_idx, test_inputs_idx, val_masks_idx, test_masks_idx, val_labels_idx, test_labels_idx = train_test_split(
    temp_inputs_idx, temp_masks_idx, temp_labels_idx, test_size=0.5, random_state=42
)

# Convert to tensors
train_inputs = tf.gather(input_ids, train_inputs_idx)
val_inputs = tf.gather(input_ids, val_inputs_idx)
test_inputs = tf.gather(input_ids, test_inputs_idx)
train_masks = tf.gather(attention_masks, train_masks_idx)
val_masks = tf.gather(attention_masks, val_masks_idx)
test_masks = tf.gather(attention_masks, test_masks_idx)
train_labels = tf.gather(labels, train_labels_idx)
val_labels = tf.gather(labels, val_labels_idx)
test_labels = tf.gather(labels, test_labels_idx)

# Convert to tf.data.Dataset
batch_size = 16
train_dataset = tf.data.Dataset.from_tensor_slices(((train_inputs, train_masks), train_labels)).shuffle(len(train_labels)).batch(batch_size)
val_dataset = tf.data.Dataset.from_tensor_slices(((val_inputs, val_masks), val_labels)).batch(batch_size)
test_dataset = tf.data.Dataset.from_tensor_slices(((test_inputs, test_masks), test_labels)).batch(batch_size)

# Load IndoBERT model from PyTorch weights
model = TFAutoModelForSequenceClassification.from_pretrained('indolem/indobert-base-uncased', from_pt=True, num_labels=len(label_encoder.classes_))

# Custom train step
@tf.function
def train_step(model, optimizer, loss_fn, accuracy_metric, x, y):
    with tf.GradientTape() as tape:
        logits = model(x, training=True).logits
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
epochs = 10
for epoch in range(epochs):
    print(f"Epoch {epoch + 1}/{epochs}")
    accuracy_metric.reset_states()
    # Training
    for step, (x_batch_train, y_batch_train) in enumerate(train_dataset):
        loss = train_step(model, optimizer, loss_fn, accuracy_metric, x_batch_train, y_batch_train)
        if step % 10 == 0:
            print(f"Training loss (for one batch) at step {step}: {loss:.4f}")
    train_accuracy = accuracy_metric.result()
    print(f"Training accuracy after epoch {epoch + 1}: {train_accuracy:.4f}")
    
    # Validation
    val_loss = 0
    accuracy_metric.reset_states()
    for x_batch_val, y_batch_val in val_dataset:
        logits = model(x_batch_val, training=False).logits
        val_loss += loss_fn(y_batch_val, logits).numpy()
        
        preds = tf.argmax(logits, axis=1, output_type=tf.int32)
        accuracy_metric.update_state(y_batch_val, preds)
    
    val_loss /= len(val_dataset)
    val_accuracy = accuracy_metric.result()
    print(f"Validation loss after epoch {epoch + 1}: {val_loss:.4f}")
    print(f"Validation accuracy after epoch {epoch + 1}: {val_accuracy:.4f}")

# Save the model
model.save_pretrained('indobert_model_epochs10_batch16')
tokenizer.save_pretrained('indobert_model_epochs10_batch16')
