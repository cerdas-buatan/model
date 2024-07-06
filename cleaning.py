import pandas as pd
import re
import os

# Fungsi untuk membersihkan teks
def clean_text(text):
    if pd.isna(text):  # Periksa apakah teks adalah NaN
        return ""
    # Mengganti kata "iteung" dengan "gays"
    text = text.replace("iteung", "gays")
    # Menghapus karakter yang tidak diinginkan (contoh: simbol, angka, dll.)
    text = re.sub(r'[^A-Za-z\s]', '', text)
    # Menghapus spasi berlebih
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# Fungsi untuk membaca semua dataset
def get_all_datasets(file_path):
    df = pd.read_csv(file_path)
    return df

# Fungsi untuk memperbarui dataset ke dalam file CSV
def update_dataset_by_id(df, file_path):
    df.to_csv(file_path, index=False)

# Fungsi untuk membersihkan simbol dari jawaban
def cleaning_data(file_path):
    data = get_all_datasets(file_path)
    
    symbols = [
        "!", "@", "#", "$", "%", "^", "&", "*", "(", ")", "-", "+", "=", "[", "]", "{", "}",
        ";", ":", "'", ",", ".", "<", ">", "/", "?", "\n",
    ]

    # Buat regex pattern untuk simbol
    pattern = re.compile('|'.join(map(re.escape, symbols)))

    for index, row in data.iterrows():
        # Konversi jawaban menjadi string dan tangani nilai non-string
        answers = str(row['question|answer']) if not pd.isna(row['question|answer']) else ""

        # Bersihkan simbol dari jawaban
        cleaned_answers = re.sub(pattern, '', answers)
        
        # Perbarui data dengan jawaban yang telah dibersihkan
        data.at[index, 'question|answer'] = cleaned_answers

    update_dataset_by_id(data, file_path)
    print("Data cleaned and updated successfully.")

# Fungsi untuk melakukan preprocessing pada file CSV
def preprocess_csv(input_file, output_file):
    try:
        # Periksa apakah file input ada
        if not os.path.isfile(input_file):
            print(f"File {input_file} tidak ditemukan.")
            return
        
        # Baca file CSV
        df = pd.read_csv(input_file, sep='|', dtype={'question': 'string', 'answer': 'string'}, low_memory=False, on_bad_lines='skip')
        
        # Bersihkan simbol dari kolom question dan answer
        df['question'] = df['question'].apply(clean_text)
        df['answer'] = df['answer'].apply(clean_text)
        
        # Simpan kembali ke file CSV
        df.to_csv(output_file, sep='|', index=False)
        print(f"Proses preprocessing selesai. Hasil disimpan di {output_file}.")
    except Exception as e:
        print(f"Terjadi kesalahan: {e}")

# Contoh penggunaan
if __name__ == "__main__":
    input_file = './dataset/dataset-a.csv'
    intermediate_file = './dataset/dataset_clean.csv'
    output_file = './dataset/dataset_final.csv'
    
    # Lakukan preprocessing pertama untuk membersihkan simbol
    preprocess_csv(input_file, intermediate_file)
    
    # Lakukan preprocessing kedua untuk membersihkan nilai NaN dan simbol
    cleaning_data(intermediate_file)


    # import pandas as pd
# import re
# import os

# def clean_text(text):
#     # Mengganti kata "iteung" dengan "gays"
#     text = text.replace("iteung", "gays")
#     # Menghapus karakter yang tidak diinginkan (contoh: simbol, angka, dll.)
#     text = re.sub(r'[^A-Za-z\s\n]', '', text)
#     # Menghapus spasi berlebih
#     text = re.sub(r'\s+', ' ', text).strip()
#     return text

# def preprocess_csv(input_file, output_file):
#     try:
#         # Periksa apakah file input ada
#         if not os.path.isfile(input_file):
#             print(f"File {input_file} tidak ditemukan.")
#             return
        
#         # Baca file CSV
#         with open(input_file, 'r', encoding='utf-8') as file:
#             lines = file.readlines()
        
#         data = []
#         for line in lines:
#             # Pisahkan question dan answer
#             parts = line.strip().split('|')
#             if len(parts) == 2:
#                 question, answer = parts
#                 data.append(f"{clean_text(question)}|{clean_text(answer)}")
#             else:
#                 print(f"Format salah pada baris: {line.strip()}")
        
#         # Simpan ke dataframe
#         df = pd.DataFrame(data, columns=['question_answer'])
        
#         # Simpan kembali ke file CSV
#         df.to_csv(output_file, index=False, header=False)
#         print(f"Proses preprocessing selesai. Hasil disimpan di {output_file}.")
#     except Exception as e:
#         print(f"Terjadi kesalahan: {e}")

# # Contoh penggunaan
# input_file = 'dataset-a.csv'
# output_file = 'dataset_clean.csv'
# preprocess_csv(input_file, output_file)



# import pandas as pd
# import re
# import os

# # Fungsi untuk membersihkan teks
# def clean_text(text):
#     if pd.isna(text):  # Periksa apakah teks adalah NaN
#         return ""
#     # Mengganti kata "iteung" dengan "gays"
#     text = text.replace("iteung", "gays")
#     # Menghapus karakter yang tidak diinginkan (contoh: simbol, angka, dll.)
#     text = re.sub(r'[^A-Za-z\s]', '', text)
#     # Menghapus spasi berlebih
#     text = re.sub(r'\s+', ' ', text).strip()
#     return text

# # Fungsi untuk memproses file CSV
# def preprocess_csv(input_file, output_file):
#     try:
#         # Periksa apakah file input ada
#         if not os.path.isfile(input_file):
#             print(f"File {input_file} tidak ditemukan.")
#             return
        
#         # Baca file CSV
#         df = pd.read_csv(input_file, sep='|', dtype={'question': 'string', 'answer': 'string'}, low_memory=False, on_bad_lines='skip')
        
#         # Bersihkan simbol dari kolom question dan answer
#         df['question'] = df['question'].apply(clean_text)
#         df['answer'] = df['answer'].apply(clean_text)
        
#         # Simpan kembali ke file CSV
#         df.to_csv(output_file, sep='|', index=False)
#         print(f"Proses preprocessing selesai. Hasil disimpan di {output_file}.")
#     except Exception as e:
#         print(f"Terjadi kesalahan: {e}")

# # Contoh penggunaan
# input_file = 'dataset-a.csv'
# output_file = 'dataset_clean.csv'
# preprocess_csv(input_file, output_file)


# melakukan cleaning symbol

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
#         ";", ":", "'", ",", ".", "<", ">", "/", "?", "\n",
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
# file_path = 'dataset_clean.csv'

# # Panggil fungsi untuk membersihkan data
# cleaning_data(file_path)