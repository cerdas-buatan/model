import pandas as pd
import tensorflow as tf
import joblib
import os

# Define folder to load model and other files
save_dir = 'saved_model'

# Load model, vectorizer, and label encoder
model = tf.keras.models.load_model(os.path.join(save_dir, 'nn_model.h5'))
vectorizer = joblib.load(os.path.join(save_dir, 'vectorizer.pkl'))
label_encoder = joblib.load(os.path.join(save_dir, 'label_encoder.pkl'))

# Predict function
def generate_answer(question):
    # Transform question to BoW
    question_bow = vectorizer.transform([question]).toarray()
    # Predict with the model
    prediction = model.predict(question_bow)
    predicted_label = tf.argmax(prediction, axis=1).numpy()[0]
    # Decode label to original text
    return label_encoder.inverse_transform([predicted_label])[0]

# Collect results
results = []

# Manual input
while True:
# while True:
    user_input = input("Masukan pertanyaan (atau 'exit' untuk keluar ): ")
    if user_input.lower() == 'exit':
        break
    predicted_answer = generate_answer(user_input)
    print(f'Jawaban: {predicted_answer}')

    # Save result to list
    results.append({'Pertanyaan': user_input, 'Jawaban': predicted_answer})

# Create a DataFrame from the results
results_df = pd.DataFrame(results)

# Save results to Excel
results_file = 'hasil_testing.xlsx'
results_df.to_excel(results_file, index=False)

print(f"Hasil testing telah di simpan '{results_file}'.")