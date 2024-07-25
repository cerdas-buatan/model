import tensorflow as tf
from transformers import TFT5ForConditionalGeneration, T5Tokenizer
import pandas as pd

#from transformers import TFT5ForConditionalGeneration, T5Tokenizer
#import pandas as pd

# load model dan tokenizer yang telah dilatih
model_path = 't5_text_to_text_model'
model = TFT5ForConditionalGeneration.from_pretrained(model_path)
tokenizer = T5Tokenizer.from_pretrained(model_path)

# Inisialisasi daftar untuk menyimpan pertanyaan, jawaban, dan akurasi
data = []
#data = []

# Fungsi menghasilkan teks dari input
def generate_text(input_text):
    try:
        # Tokenisasi input
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

##
    # Dalam kasus nyata, ini harus membandingkan teks yang dihasilkan dengan jawaban yang diharapkan.
    return "Akurasi Placeholder"

    # Dalam kasus nyata, ini harus membandingkan teks yang dihasilkan dengan jawaban yang diharapkan.
    # return "Akurasi Placeholder"

# Penggunaan
while True:
    input_text = input("Masukkan pertanyaan Anda (atau ketik 'exit' untuk keluar): ")

    if input_text.lower() == 'exit':
        break

    # Menghasilkan teks jawaban berdasarkan input
    generated_text = generate_text(input_text)
    print("Jawaban dari model:")
    print(generated_text)
    
    # Mengevaluasi akurasi
    accuracy = evaluate_accuracy(input_text, generated_text)


    # Mengevaluasi akurasi
    # accuracy = evaluate_accuracy(input_text, generated_text)
##

    # Menyimpan pertanyaan, jawaban, dan akurasi ke dalam daftar data
    data.append({"Pertanyaan": input_text, "Jawaban": generated_text, "Akurasi": accuracy})

#
# Menyimpan data testing ke file Excel
df = pd.DataFrame(data)
df.to_excel("hasil_test.xlsx", index=False)
print("Data berhasil disimpan ke hasil_test.xlsx")
