import tensorflow as tf
from transformers import GPT2Tokenizer, TFGPT2LMHeadModel
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
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
tokenizer.pad_token = tokenizer.eos_token  # Ensure that the padding token is set to the EOS token

# Tokenize the input and output sequences
input_ids = []
attention_masks = []

for index, row in df.iterrows():
    encoded_input = tokenizer.encode_plus(row['question'] + tokenizer.eos_token + row['answer'], 
                                          add_special_tokens=True, 
                                          max_length=128, 
                                          padding='max_length', 
                                          return_attention_mask=True, 
                                          truncation=True)

    input_ids.append(encoded_input['input_ids'])
    attention_masks.append(encoded_input['attention_mask'])

input_ids = tf.constant(input_ids)
attention_masks = tf.constant(attention_masks)

# Load the GPT-2 model
model = TFGPT2LMHeadModel.from_pretrained('gpt2')

# Define the optimizer and loss function
optimizer = tf.keras.optimizers.Adam(learning_rate=5e-5)
loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

# Compile the model
model.compile(optimizer=optimizer, loss=loss, metrics=['accuracy'])

# Train the model
model.fit([input_ids, attention_masks], input_ids, epochs=10, batch_size=1)

# Save the model and tokenizer
model_path = 'gpt2_text_to_text_model'
model.save_pretrained(model_path)
tokenizer.save_pretrained(model_path)
