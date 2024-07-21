import tensorflow as tf
from transformers import TFT5ForConditionalGeneration, T5Tokenizer
import pandas as pd

# load model dan tokenizer yang telah dilatih
model_path = 't5_text_to_text_model'
model = TFT5ForConditionalGeneration.from_pretrained(model_path)
tokenizer = T5Tokenizer.from_pretrained(model_path)

# Inisialisasi daftar untuk menyimpan pertanyaan, jawaban, dan akurasi
data = []

# Fungsi menghasilkan teks dari input
def generate_text(input_text):
    try:
        # Tokenisasi input
        inputs = tokenizer.encode_plus(input_text, return_tensors='tf', add_special_tokens=True, max_length=64, padding='max_length', truncation=True)
        input_ids = inputs['input_ids']
        attention_mask = inputs['attention_mask']

         # Menghasilkan teks dari model
        outputs = model.generate(input_ids, attention_mask=attention_mask, max_length=64, num_beams=4, early_stopping=True)

  # dekode output yang dihasilkan
        output_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

        return output_text
    except Exception as e:
        return f"Error dalam menghasilkan teks: {str(e)}"
    

# Fungsi mengevaluasi akurasi teks yang dihasilkan (placeholder untuk implementasi sebenarnya)
def evaluate_accuracy(input_text, generated_text):
    # Untuk kesederhanaan, kita mengembalikan nilai akurasi placeholder.
    # Dalam kasus nyata, ini harus membandingkan teks yang dihasilkan dengan jawaban yang diharapkan.
    return "Akurasi Placeholder"

# Contoh input untuk evaluasi
example_inputs = [
    "Apa itu machine learning?",
    "Bagaimana cara kerja jaringan syaraf tiruan?",
    "Jelaskan konsep pengelompokan data dalam data mining."
]

# Evaluasi beberapa contoh input
for input_text in example_inputs:
    generated_text = generate_text(input_text)
    accuracy = evaluate_accuracy(input_text, generated_text)
    data.append({'question': input_text, 'generated_answer': generated_text, 'accuracy': accuracy})

# Simpan hasil evaluasi ke file CSV
df = pd.DataFrame(data)
df.to_csv('evaluation_results.csv', index=False)

# Cetak hasil evaluasi
for row in data:
    print(f"Question: {row['question']}")
    print(f"Generated Answer: {row['generated_answer']}")
    print(f"Accuracy: {row['accuracy']}")
    print("-" * 50)