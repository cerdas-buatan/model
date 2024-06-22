import pandas as pd
import re
import csv

# Load dataset
# import csv

rows = []
with open('aset.csv', 'r', encoding='utf-8') as f:
    csv_reader = csv.reader(f, delimiter='|')
    for row in csv_reader:
        if len(row) == 2:  # Pastikan hanya mengambil baris dengan 2 kolom
            rows.append({'question': row[0], 'answer': row[1]})

df = pd.DataFrame(rows)

# Function to normalize and validate data
def normalize_and_validate(df):
    # Normalize text function
    def normalize_text(text):
        text = text.lower()
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        text = re.sub(r'\n+', ' ', text).strip()
        return text
    
    # Normalize question and answer columns
    df['question'] = df['question'].apply(normalize_text)
    df['answer'] = df['answer'].apply(normalize_text)
    
    # Remove rows with empty or NaN values
    df.dropna(inplace=True)
    df.drop(df[df['question'] == ''].index, inplace=True)
    df.drop(df[df['answer'] == ''].index, inplace=True)
    
    return df

# Normalize and validate dataset
df_validated = normalize_and_validate(df)

# Print information about the dataset after validation
print(f"Original dataset length: {len(df)}")
print(f"Validated dataset length: {len(df_validated)}")
