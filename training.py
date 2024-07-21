# import library yang di butuhkan
import tensorflow as tf
from transformers import TFT5ForConditionalGeneration, T5Tokenizer
import pandas as pd 

# Inisialisasi daftar kosong untuk menyimpan baris yang telah dibersihkan
rows = []

# Read and clean dataset, handling any anomalies
with open('dataset_clean2.csv', 'r', encoding='utf-8') as file:
    for line_number, line in enumerate(file):
        # Pisahkan baris berdasarkan '|' dan tangani baris yang tidak terduga
        parts = line.strip().split('|')
        if len(parts) == 2:  # Hanya memproses baris dengan tepat dua bagian
            rows.append(parts)

# Konversi baris yang telah dibersihkan ke DataFrame
df = pd.DataFrame(rows, columns=['question', 'answer'])

# Mengatasi missing values
df.dropna(inplace=True)

# Inisialisasi tokenizer
tokenizer = T5Tokenizer.from_pretrained('t5-small')

# Tokenisasi input dan output sequence
input_ids = []
attention_masks = []
labels = []

for index, row in df.iterrows():
    encoded_input = tokenizer.encode_plus(row['question'], add_special_tokens=True, max_length=64, padding='max_length', return_attention_mask=True, truncation=True)
    encoded_output = tokenizer.encode_plus(row['answer'], add_special_tokens=True, max_length=64, padding='max_length', return_attention_mask=True, truncation=True)

    input_ids.append(encoded_input['input_ids'])
    attention_masks.append(encoded_input['attention_mask'])
    # Geser label ke kanan untuk model
    label_ids = encoded_output['input_ids']
    label_ids = [tokenizer.pad_token_id] + label_ids[:-1]  # Geser ke kanan
    labels.append(label_ids)

input_ids = tf.constant(input_ids)
attention_masks = tf.constant(attention_masks)
labels = tf.constant(labels)

# Muat model sequence-to-sequence
model = TFT5ForConditionalGeneration.from_pretrained('t5-small')

# Definisikan optimizer
optimizer = tf.keras.optimizers.Adam(learning_rate=5e-5)

# Definisikan fungsi loss custom
def compute_loss(labels, logits):
    return tf.keras.losses.sparse_categorical_crossentropy(labels, logits, from_logits=True)

# Compile model
model.compile(optimizer=optimizer, loss=compute_loss) 

# Definisikan dataset untuk training 
dataset = tf.data.Dataset.from_tensor_slices(({"input_ids": input_ids, "attention_mask": attention_masks}, labels))
dataset = dataset.shuffle(buffer_size=1024).batch(10) 

# Training loop 
epochs = 3   
for epoch in range(epochs):     
    print(f'Epoch {epoch + 1}/{epochs}') 
    for batch in dataset: 
        inputs, targets = batch 
        with tf.GradientTape() as tape: 
            outputs = model(inputs, labels=targets) 
            loss = outputs.loss 
        gradients = tape.gradient(loss, model.trainable_variables) 
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))
        print(f'Loss: {loss.numpy()}') 

# Evaluasi model
def evaluate_model(model, tokenizer, question, answer):
    input_ids = tokenizer.encode(question, return_tensors='tf')
    generated_ids = model.generate(input_ids, max_length=64, num_beams=4, early_stopping=True)
    generated_answer = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
    print(f'Question: {question}')
    print(f'Expected Answer: {answer}')
    print(f'Generated Answer: {generated_answer}')

# Uji model dengan contoh
test_question = "Apa itu machine learning?"
test_answer = "Machine learning adalah cabang dari kecerdasan buatan yang berfokus pada pengembangan algoritma yang memungkinkan komputer belajar dari data."
evaluate_model(model, tokenizer, test_question, test_answer)