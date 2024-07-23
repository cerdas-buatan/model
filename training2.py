# import yang di butuhkan
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer

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
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')

# Tokenisasi input dan output sequence
input_ids = []
attention_masks = []
labels = []

for index, row in df.iterrows():
    encoded_input = tokenizer.encode(row['question'], add_special_tokens=True, max_length=64, padding='max_length', truncation=True)
    encoded_output = tokenizer.encode(row['answer'], add_special_tokens=True, max_length=64, padding='max_length', truncation=True)

    input_ids.append(encoded_input)
    labels.append(encoded_output)

# Padding untuk input_ids dan labels
max_length = max(len(ids) for ids in input_ids)
input_ids = tf.keras.preprocessing.sequence.pad_sequences(input_ids, maxlen=max_length, padding='post')
labels = tf.keras.preprocessing.sequence.pad_sequences(labels, maxlen=max_length, padding='post')

# Konversi ke TensorFlow tensor
input_ids = tf.constant(input_ids)
labels = tf.constant(labels)

# Muat model GPT-2
model = TFGPT2LMHeadModel.from_pretrained('gpt2')

# Definisikan optimizer
optimizer = tf.keras.optimizers.Adam(learning_rate=5e-5)

# Definisikan fungsi loss custom
def compute_loss(labels, logits):
    return tf.keras.losses.sparse_categorical_crossentropy(labels, logits, from_logits=True)

# Definisikan fungsi akurasi custom
def masked_accuracy(y_true, y_pred):
    y_true = tf.cast(tf.reshape(y_true, (-1,)), tf.int64)
    y_pred = tf.cast(tf.argmax(y_pred, axis=-1), tf.int64)
    y_pred = tf.reshape(y_pred, (-1,))  # Pastikan y_pred diubah bentuknya agar sesuai dengan y_true
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
model_path = 'gpt2_model'
model.save_pretrained(model_path)
tokenizer.save_pretrained(model_path)