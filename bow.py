# import yang di butuhkan
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer

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

# Inisialisasi CountVectorizer
vectorizer = CountVectorizer()

# Gabungkan kolom 'question' dan 'answer' untuk representasi BoW
combined_texts = df['question'] + " " + df['answer']

# Fit dan transform teks ke BoW
bow_matrix = vectorizer.fit_transform(combined_texts)

# Konversi BoW matrix ke DataFrame
bow_df = pd.DataFrame(bow_matrix.toarray(), columns=vectorizer.get_feature_names_out())

# Tampilkan beberapa baris dari DataFrame BoW
print(bow_df.head())

# Simpan BoW dataframe ke file CSV jika diperlukan
bow_df.to_csv('bow_representation.csv', index=False)