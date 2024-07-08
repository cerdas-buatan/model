import tensorflow as tf
from transformers import TFT5ForConditionalGeneration, T5Tokenizer
import pandas as pd

# Initialize an empty list to store cleaned rows
rows = []

# Read and clean the dataset, handling any anomalies
with open('./dataset/dataset_sample.csv', 'r', encoding='utf-8') as file:
    for line_number, line in enumerate(file):
        # Split line by '|' and handle any unexpected lines
        parts = line.strip().split('|')
        if len(parts) == 2:  # Only process lines with exactly two parts
            rows.append(parts)

# Convert cleaned rows to a DataFrame
df = pd.DataFrame(rows, columns=['question', 'answer'])

# Initialize the tokenizer
tokenizer = T5Tokenizer.from_pretrained('t5-small', legacy=False)

# Tokenize the input and output sequences
input_ids = []
attention_masks = []
decoder_input_ids = []
decoder_attention_masks = []

for index, row in df.iterrows():
    encoded_input = tokenizer.encode_plus(row['question'], add_special_tokens=True, max_length=64, padding='max_length', return_attention_mask=True, truncation=True)
    encoded_output = tokenizer.encode_plus(row['answer'], add_special_tokens=True, max_length=64, padding='max_length', return_attention_mask=True, truncation=True)

    input_ids.append(encoded_input['input_ids'])
    attention_masks.append(encoded_input['attention_mask'])
    decoder_input_ids.append(encoded_output['input_ids'])
    decoder_attention_masks.append(encoded_output['attention_mask'])

input_ids = tf.constant(input_ids)
attention_masks = tf.constant(attention_masks)
decoder_input_ids = tf.constant(decoder_input_ids)
decoder_attention_masks = tf.constant(decoder_attention_masks)

# Load the sequence-to-sequence model
model = TFT5ForConditionalGeneration.from_pretrained('t5-small')

# Define the optimizer
optimizer = tf.keras.optimizers.Adam(learning_rate=5e-5)

# Define the loss function
def custom_loss(y_true, y_pred):
    y_true = y_true[:, 1:]  # Shift left the true labels
    y_pred = y_pred[:, :-1]  # Remove the last token prediction
    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    return loss_fn(y_true, y_pred)

# Compile the model
model.compile(optimizer=optimizer, loss=custom_loss)

# Prepare the dataset
dataset = tf.data.Dataset.from_tensor_slices((
    {'input_ids': input_ids, 'attention_mask': attention_masks, 'decoder_input_ids': decoder_input_ids, 'decoder_attention_mask': decoder_attention_masks},
    decoder_input_ids
))

# Batch and shuffle the dataset
batch_size = 8
dataset = dataset.shuffle(buffer_size=1024).batch(batch_size)

# Train the model
model.fit(
    dataset,
    epochs=10
)

# Save the model and tokenizer
model_path = 't5_text_to_text_model'
model.save_pretrained(model_path)
tokenizer.save_pretrained(model_path)
