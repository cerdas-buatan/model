# import pandas as pd
# import re

# # Fungsi untuk mendapatkan semua dataset
# def get_all_datasets(file_path):
#     df = pd.read_csv(file_path)
#     return df

# # Fungsi untuk memperbarui dataset berdasarkan ID
# def update_dataset_by_id(df, file_path):
#     df.to_csv(file_path, index=False)

# # Fungsi untuk membersihkan simbol dari jawaban
# def cleaning_data(file_path):
#     data = get_all_datasets(file_path)
    
#     symbols = [
#         "!", "@", "#", "$", "%", "^", "&", "*", "(", ")", "-", "+", "=", "[", "]", "{", "}",
#         ";", ":", "'", "|", ",", ".", "<", ">", "/", "?", "\n", ","
#     ]

#     # Buat regex pattern untuk simbol
#     pattern = re.compile('|'.join(map(re.escape, symbols)))

#     for index, row in data.iterrows():
#         # Konversi jawaban menjadi string dan tangani nilai non-string
#         answers = str(row['question|answer']) if not pd.isna(row['question|answer']) else ""

#         # Bersihkan simbol dari jawaban
#         cleaned_answers = re.sub(pattern, '', answers)
        
#         # Perbarui data dengan jawaban yang telah dibersihkan
#         data.at[index, 'question|answer'] = cleaned_answers

#     update_dataset_by_id(data, file_path)
#     print("Data cleaned and updated successfully.")

# # Path ke file dataset CSV
# file_path = 'dataset-a.csv'

# # Panggil fungsi untuk membersihkan data
# cleaning_data(file_path)


import pandas as pd
import re

# Fungsi untuk membaca semua dataset dengan dtype yang sesuai
def get_all_datasets(file_path):
    # Tentukan dtype kolom question dan answer sebagai string
    dtype = {
        'question|answer': 'string'
    }
    df = pd.read_csv(file_path, sep='|', dtype=dtype, low_memory=False,error_bad_lines=False)
    return df

# Fungsi untuk memperbarui dataset ke dalam file CSV
def update_dataset_by_id(df, file_path):
    df.to_csv(file_path, sep='|', index=False)

# Fungsi untuk membersihkan simbol dari teks
def clean_text(text):
    symbols = [
        "!", "@", "#", "$", "%", "^", "&", "*", "(", ")", "-", "+", "=", "[", "]", "{", "}",
        ";", ":", "'", "|", ",", ".", "<", ">", "/", "?", "\n"
    ]
    # Buat regex pattern untuk simbol
    pattern = re.compile('|'.join(map(re.escape, symbols)))
    # Konversi teks menjadi string jika tidak kosong atau NaN
    if pd.isna(text):
        return ""
    return re.sub(pattern, '', str(text))

# Fungsi untuk membersihkan data dalam kolom question dan answer
def cleaning_data(file_path):
    data = get_all_datasets(file_path)
    
    # Bersihkan simbol dari kolom question|answer
    data['question|answer'] = data['question|answer'].apply(clean_text)
    
    update_dataset_by_id(data, file_path)
    print("Data cleaned and updated successfully.")

# Path ke file dataset CSV
file_path = 'dataset-a.csv'

# Panggil fungsi untuk membersihkan data
cleaning_data(file_path)

