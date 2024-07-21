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



