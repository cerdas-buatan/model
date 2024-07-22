# import yg di butuhkan
import tensorflow as tf
from transformers import TFGPT2LMHeadModel, GPT2Tokenizer
import pandas as pd

# Inisialisasi daftar kosong untuk menyimpan baris yang telah dibersihkan
rows = []

# Read and clean dataset, handling any anomalies
with open('dataset_clean2.csv', 'r', encoding='utf-8') as file:
    for line_number, line in enumerate(file):
        # Pisahkan baris berdasarkan '|' dan tangani baris yang tidak terduga
        parts = line.strip().split('|')
        if len(parts) == 2:  # Hanya memproses baris dengan tepat dua bagian
            rows.append(parts)

# Konversi baris yang telah dibersihkan ke DataFrame
df = pd.DataFrame(rows, columns=['question', 'answer'])

# Mengatasi missing values
df.dropna(inplace=True)

# Inisialisasi tokenizer
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')

# Tokenisasi input dan output sequence
input_ids = []
attention_masks = []
labels = []

for index, row in df.iterrows():
    encoded_input = tokenizer.encode(row['question'], add_special_tokens=True, max_length=64, padding='max_length', truncation=True)
    encoded_output = tokenizer.encode(row['answer'], add_special_tokens=True, max_length=64, padding='max_length', truncation=True)

    input_ids.append(encoded_input)
    labels.append(encoded_output)

