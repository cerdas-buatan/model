import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import tensorflow as tf
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
train_dataset = tf.data.Dataset.from_tensor_slices((X_train_bow.toarray(), y_train)).batch(64)
test_dataset = tf.data.Dataset.from_tensor_slices((X_test_bow.toarray(), y_test)).batch(64)


# Definisikan model Neural Network
model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(X_train_bow.shape[1],)),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(len(label_encoder.classes_), activation='softmax')
])

# Kompilasi model
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Latih model
model.fit(train_dataset, epochs=100)

# Simpan model, vectorizer, dan label encoder di folder yang ditentukan
model.save(os.path.join(save_dir, 'nn_model.h5'))
joblib.dump(vectorizer, os.path.join(save_dir, 'vectorizer.pkl'))
joblib.dump(label_encoder, os.path.join(save_dir, 'label_encoder.pkl'))

print(f"Training complete. Model, vectorizer, and label encoder saved in '{save_dir}'.")