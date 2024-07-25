import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import tensorflow as tf
import joblib
import os

# Inisialisasi daftar kosong untuk menyimpan baris yang telah dibersihkan
rows = []

# Read and clean dataset, handling any anomalies
with open('dataset_clean2.csv', 'r', encoding='utf-8') as file:
    for line_number, line in enumerate(file):
        # Pisahkan baris berdasarkan '|'
        parts = line.strip().split('|')
        if len(parts) == 2:  # Hanya memproses baris dengan tepat dua bagian
            rows.append(parts)

# Konversi baris yang telah dibersihkan ke DataFrame
df = pd.DataFrame(rows, columns=['question', 'answer'])

# Mengatasi missing values
df.dropna(inplace=True)

# Inisialisasi CountVectorizer
vectorizer = CountVectorizer()

# Fit dan transform teks ke BoW
X_bow = vectorizer.fit_transform(df['question'])

# Menggunakan kolom 'answer' sebagai label
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(df['answer'])

# Bagi dataset menjadi training dan testing set (80% train, 20% test)
X_train_bow, X_test_bow, y_train, y_test = train_test_split(X_bow, y, test_size=0.2, random_state=42)

# Konversi data ke TensorFlow dataset
train_dataset = tf.data.Dataset.from_tensor_slices((X_train_bow.toarray(), y_train)).batch(10)
test_dataset = tf.data.Dataset.from_tensor_slices((X_test_bow.toarray(), y_test)).batch(10)

