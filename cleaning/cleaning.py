import pandas as pd
import re
import os

# Function to clean text
def clean_text(text):
    if pd.isna(text):  # Check if text is NaN
        return ""
    # Remove specific patterns (e.g., "nn" and "nnterimakasih")
    text = re.sub(r'\bnn\b', '', text)  # Remove standalone "nn"
    text = re.sub(r'n+n+', '', text)  # Remove sequences like "nnterimakasih"
    # # Replace the word "iteung" with "gays"
    text = text.replace("iteung", "disal")
    # # Remove unwanted characters (e.g., symbols, numbers, etc.)
    # text = re.sub(r'[^A-Za-z\s]', '', text)
    # Remove extra spaces
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# Function to preprocess and clean the dataset
def preprocess_csv(input_file, output_file):
    try:
        # Check if the input file exists
        if not os.path.isfile(input_file):
            print(f"File {input_file} not found.")
            return
        
        # Load the dataset from the file
        df = pd.read_csv(input_file, sep='|', dtype={'question': 'string', 'answer': 'string'}, low_memory=False, on_bad_lines='skip')
        
        # Clean the text in 'question' and 'answer' columns
        df['question'] = df['question'].apply(clean_text)
        df['answer'] = df['answer'].apply(clean_text)
        
        # Remove rows with missing 'question' or 'answer'
        df = df.dropna(subset=['question', 'answer'])
        
        # Remove duplicate rows
        df = df.drop_duplicates()
        
        # Optionally remove rows with empty 'question' or 'answer'
        df = df[(df['question'] != '') & (df['answer'] != '')]
        
        # Save the cleaned data to a new CSV file
        df.to_csv(output_file, sep='|', index=False)
        print(f"Preprocessing completed. Results saved to {output_file}.")
    except Exception as e:
        print(f"An error occurred: {e}")

# Example usage
input_file = 'data_clean.csv'  # Input file with raw data
output_file = 'data_cleaned.csv'  # Output file for cleaned data
preprocess_csv(input_file, output_file)
