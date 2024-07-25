import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import tensorflow as tf
import joblib
import os
from transformers import BertTokenizer, TFBertModel

# Define folder to save model and other files
save_dir = 'save_model'
os.makedirs(save_dir, exist_ok=True)

# Inisialisasi daftar kosong untuk menyimpan baris yang telah dibersihkan
rows = []

# Read and clean dataset, handling any anomalies
with open('data.csv', 'r', encoding='utf-8') as file:
    for line_number, line in enumerate(file):
        # Pisahkan baris berdasarkan '|'
        parts = line.strip().split('|')
        if len(parts) == 2:  # Hanya memproses baris dengan tepat dua bagian
            rows.append(parts)

# Konversi baris yang telah dibersihkan ke DataFrame
df = pd.DataFrame(rows, columns=['question', 'answer'])

# Mengatasi missing values
df.dropna(inplace=True)

# Inisialisasi BERT tokenizer dan model
tokenizer = BertTokenizer.from_pretrained('indobenchmark/indobert-base-p2')
bert_model = TFBertModel.from_pretrained('indobenchmark/indobert-base-p2')

# Fungsi untuk mengubah teks menjadi BERT embeddings
def encode_text(texts):
    input_ids = tokenizer(texts.tolist(), return_tensors='tf', padding=True, truncation=True)['input_ids']
    return bert_model(input_ids)[0][:, 0, :].numpy()

# Encode pertanyaan menggunakan IndoBERT
X_bert = encode_text(df['question'])

# Menggunakan kolom 'answer' sebagai label
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(df['answer'])

# Bagi dataset menjadi training dan testing set (80% train, 20% test)
X_train_bert, X_test_bert, y_train, y_test = train_test_split(X_bert, y, test_size=0.2, random_state=42)

# Konversi data ke TensorFlow dataset
train_dataset = tf.data.Dataset.from_tensor_slices((X_train_bert, y_train)).batch(64)
test_dataset = tf.data.Dataset.from_tensor_slices((X_test_bert, y_test)).batch(64)

# Definisikan model Neural Network
model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(X_train_bert.shape[1],)),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(len(label_encoder.classes_), activation='softmax')
])

# Kompilasi model
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Latih model
model.fit(train_dataset, epochs=50)

# Simpan model dan label encoder di folder yang ditentukan
model.save(os.path.join(save_dir, 'nn_model.h5'))
joblib.dump(label_encoder, os.path.join(save_dir, 'label_encoder.pkl'))

print(f"Training complete. Model and label encoder saved in '{save_dir}'.")
