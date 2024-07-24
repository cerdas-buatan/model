# import library yang di butuhkan
import tensorflow as tf
from transformers import TFT5ForConditionalGeneration, T5Tokenizer
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

# Inisialisasi tokenizer
tokenizer = T5Tokenizer.from_pretrained('t5-small')

# Tokenisasi input dan output sequence
input_ids = []
attention_masks = []
labels = []

for index, row in df.iterrows():
    encoded_input = tokenizer.encode_plus(row['question'], add_special_tokens=True, max_length=64, padding='max_length', return_attention_mask=True, truncation=True)
    encoded_output = tokenizer.encode_plus(row['answer'], add_special_tokens=True, max_length=64, padding='max_length', return_attention_mask=True, truncation=True)

    input_ids.append(encoded_input['input_ids'])
    attention_masks.append(encoded_input['attention_mask'])

    # Geser label ke kanan untuk model
    label_ids = encoded_output['input_ids']
    label_ids = [tokenizer.pad_token_id] + label_ids[:-1]  # Geser ke kanan
    labels.append(label_ids)