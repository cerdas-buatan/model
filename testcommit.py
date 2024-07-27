import pandas as pd
import tensorflow as tf
from transformers import DistilBertTokenizer, TFDistilBertModel
import joblib
import os

# Define folder to load model and other files
save_dir = 'save_model'

# Load tokenizer and label encoder
tokenizer = DistilBertTokenizer.from_pretrained(save_dir)
label_encoder = joblib.load(os.path.join(save_dir, 'label_encoder.pkl'))

# Load model with custom objects
with tf.keras.utils.custom_object_scope({'TFDistilBertModel': TFDistilBertModel}):
    model = tf.keras.models.load_model(os.path.join(save_dir, 'best_model.h5'))

# Predict function
def generate_answer(question):
    # Tokenize the question
    inputs = tokenizer(question, return_tensors='tf', padding='max_length', truncation=True, max_length=128)
    input_ids = inputs['input_ids']
    attention_mask = inputs['attention_mask']
    # Predict with the model
    prediction = model.predict({'input_ids': input_ids, 'attention_mask': attention_mask})
    predicted_label = tf.argmax(prediction, axis=1).numpy()[0]
    # Decode label to original text
    return label_encoder.inverse_transform([predicted_label])[0]

# Collect results
results = []

# Manual input
while True:
    user_input = input("Masukan pertanyaan (atau 'exit' untuk keluar): ")
    if user_input.lower() == 'exit':
        break
    predicted_answer = generate_answer(user_input)
    print(f'Jawaban: {predicted_answer}')

    # Save result to list
    results.append({'Pertanyaan': user_input, 'Jawaban': predicted_answer})

# Create a DataFrame from the results
results_df = pd.DataFrame(results)

# Save results to Excel
results_file = 'result_testing.xlsx'
results_df.to_excel(results_file, index=False)
print(f"Hasil testing telah disimpan '{results_file}'.")
