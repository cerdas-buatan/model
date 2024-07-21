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


def main():
    model_path = 't5_text_to_text_model'
    bert_model_name = 'indolem/indobert-base-uncased'
    dataset_path = 'dataset_clean2.csv'

    # Load models and tokenizers
    model_t5, tokenizer_t5, model_bert, tokenizer_bert = load_model_and_tokenizer(model_path, bert_model_name)

    # Load and preprocess data
    df, label_encoder = load_and_preprocess_data(dataset_path)

    # Example usage
    input_text = "Apa ibu kota Indonesia?"
    output_text = generate_text(input_text, tokenizer_t5, model_t5)
    
    print(f"Input: {input_text}")
    print(f"Output: {output_text}")

if __name__ == "__main__":
    main()
