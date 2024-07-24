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

# Definisikan fungsi loss custom
def compute_loss(labels, logits):
    return tf.keras.losses.sparse_categorical_crossentropy(labels, logits, from_logits=True)
# Definisikan fungsi akurasi custom
def masked_accuracy(y_true, y_pred):
    y_true = tf.cast(tf.reshape(y_true, (-1,)), tf.int64)
    y_pred = tf.cast(tf.argmax(y_pred, axis=-1), tf.int64)
    y_pred = tf.reshape(y_pred, (-1,)) 
# Pastikan y_pred diubah bentuknya agar sesuai dengan y_true
    accuracy = tf.equal(y_true, y_pred)
    mask = tf.cast(tf.not_equal(y_true, tokenizer.pad_token_id), tf.float32)  # Abaikan token padding
    accuracy = tf.cast(accuracy, tf.float32) * mask
    return tf.reduce_sum(accuracy) / tf.reduce_sum(mask)

# Kompilasi model dengan metrik akurasi custom
model.compile(optimizer=optimizer, loss=compute_loss, metrics=[masked_accuracy])

# Buat tf.data.Dataset
dataset = tf.data.Dataset.from_tensor_slices((
    {
        'input_ids': input_ids,
        'attention_mask': attention_masks,
        'labels': labels
    },
    labels
)).batch(5)

# Train model for more epochs
model.fit(dataset, epochs=100)

# Simpan model dan tokenizer
model_path = 't5_text_to_text_model'
model.save_pretrained(model_path)
tokenizer.save_pretrained(model_path)

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