# import yang di butuhkan
import pandas as pd
import tensorflow as tf
import joblib
import os

# Tentukan folder untuk memuat model dan file lainnya
save_dir = 'saved_model'

# Muat model, vektorizer, dan label encoder
model = tf.keras.models.load_model(os.path.join(save_dir, 'nn_model.h5'))
vectorizer = joblib.load(os.path.join(save_dir, 'vectorizer.pkl'))
label_encoder = joblib.load(os.path.join(save_dir, 'label_encoder.pkl'))

# Fungsi untuk menghasilkan jawaban
def generate_answer(question):
    # Transformasi pertanyaan ke BoW
    question_bow = vectorizer.transform([question]).toarray()
    # Prediksi dengan model
    prediction = model.predict(question_bow)
    predicted_label = tf.argmax(prediction, axis=1).numpy()[0]
    # Decode label ke teks asli
    return label_encoder.inverse_transform([predicted_label])[0]

# Kumpulkan hasil
results = []

# Input manual
while True:
    user_input = input("Masukkan pertanyaan (atau 'exit' untuk keluar): ")
    if user_input.lower() == 'exit':
        break
    predicted_answer = generate_answer(user_input)
    print(f'Jawaban: {predicted_answer}')

    # Simpan hasil ke dalam daftar
    results.append({'Pertanyaan': user_input, 'Jawaban': predicted_answer})

# Buat DataFrame dari hasil
results_df = pd.DataFrame(results)

# Simpan hasil ke Excel
results_file = 'hasil_testing.xlsx'
results_df.to_excel(results_file, index=False)

print(f"Hasil testing telah disimpan di '{results_file}'.")
