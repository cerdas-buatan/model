import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import tensorflow as tf
from transformers import BertTokenizer, TFBertForSequenceClassification
import joblib
import os

# Define folder to save model and other files
save_dir = 'saved_model'
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

# Initialize BertTokenizer
tokenizer = BertTokenizer.from_pretrained('indobenchmark/indobert-base-p2')

# Tokenize and encode the questions
X = tokenizer(
    df['question'].tolist(), 
    padding=True, 
    truncation=True, 
    return_tensors='tf'
)

# Menggunakan kolom 'answer' sebagai label
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(df['answer'])

# Bagi dataset menjadi training dan testing set (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Konversi data ke TensorFlow dataset
train_dataset = tf.data.Dataset.from_tensor_slices((dict(X_train), y_train)).batch(64)
test_dataset = tf.data.Dataset.from_tensor_slices((dict(X_test), y_test)).batch(64)

# Load pre-trained IndoBERT model for sequence classification
model = TFBertForSequenceClassification.from_pretrained('indobenchmark/indobert-base-p2', num_labels=len(label_encoder.classes_))

# Kompilasi model
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=5e-5),
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

# Latih model
model.fit(train_dataset, epochs=3)  # IndoBERT biasanya membutuhkan lebih sedikit epoch

# Simpan model, tokenizer, dan label encoder di folder yang ditentukan
model.save_pretrained(save_dir)
tokenizer.save_pretrained(save_dir)
joblib.dump(label_encoder, os.path.join(save_dir, 'label_encoder.pkl'))

print(f"Training complete. Model, tokenizer, and label encoder saved in '{save_dir}'.")
