import tensorflow as tf
from transformers import GPT2Tokenizer, TFGPT2LMHeadModel
import pandas as pd
from sklearn.model_selection import train_test_split

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

# Split the dataset into training and testing sets
train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)

# Initialize the tokenizer
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
tokenizer.pad_token = tokenizer.eos_token  # Ensure that the padding token is set to the EOS token

# Tokenize the input and output sequences
def tokenize_data(df):
    input_ids = []
    attention_masks = []

    for index, row in df.iterrows():
        # Concatenate question and answer with EOS token
        encoded_input = tokenizer.encode_plus(row['question'] + tokenizer.eos_token + row['answer'], 
                                              add_special_tokens=True, 
                                              max_length=128, 
                                              padding='max_length', 
                                              return_attention_mask=True, 
                                              truncation=True)
        input_ids.append(encoded_input['input_ids'])
        attention_masks.append(encoded_input['attention_mask'])

    return tf.constant(input_ids), tf.constant(attention_masks)

train_input_ids, train_attention_masks = tokenize_data(train_df)

# Load the GPT-2 model
model = TFGPT2LMHeadModel.from_pretrained('gpt2')

# Define the optimizer and loss function
optimizer = tf.keras.optimizers.Adam(learning_rate=5e-5)
loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

# Compile the model
model.compile(optimizer=optimizer, loss=[loss], metrics=['accuracy'])

# Train the model
model.fit({'input_ids': train_input_ids, 'attention_mask': train_attention_masks}, train_input_ids, epochs=5, batch_size=5)

# Save the model and tokenizer
model_path = 'gpt2_text_to_text_model'
model.save_pretrained(model_path)
tokenizer.save_pretrained(model_path)
