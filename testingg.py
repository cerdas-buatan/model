import tensorflow as tf
from transformers import TFT5ForConditionalGeneration, T5Tokenizer, AutoTokenizer, TFAutoModelForSequenceClassification
import pandas as pd
import csv
from sklearn.preprocessing import LabelEncoder

# Load the trained model and tokenizer
model_path = 't5_text_to_text_model'

try:
    model_t5 = TFT5ForConditionalGeneration.from_pretrained(model_path)
    tokenizer_t5 = T5Tokenizer.from_pretrained(model_path)
except Exception as e:
    print(f"Error loading T5 model or tokenizer: {str(e)}")
    exit()

# Load the tokenizer and model for IndoBERT
try:
    tokenizer_bert = AutoTokenizer.from_pretrained('indolem/indobert-base-uncased')
    model_bert = TFAutoModelForSequenceClassification.from_pretrained('indolem/indobert-base-uncased', from_pt=True)
except Exception as e:
    print(f"Error loading IndoBERT model or tokenizer: {str(e)}")
    exit()

# Load the dataset and encode labels
with open('dataset_clean2.csv', 'r', encoding='utf-8') as file:
    reader = csv.reader(file, delimiter='|')
    filtered_rows = [row for row in reader if len(row) == 2 and row[0].strip() != "" and row[1].strip() != ""]

df = pd.DataFrame(filtered_rows, columns=['question', 'answer'])
label_encoder = LabelEncoder()
label_encoder.fit(df['answer'])

# Function to generate text from input
def generate_text(input_text):
    try:
        # Tokenize the input text using T5 tokenizer
        inputs = tokenizer_t5.encode_plus(input_text, return_tensors='tf', add_special_tokens=True, max_length=50, padding='max_length', truncation=True)
        input_ids = inputs['input_ids']
        attention_mask = inputs['attention_mask']

        # Generate text from the T5 model
        outputs = model_t5.generate(input_ids, attention_mask=attention_mask, max_length=50, num_beams=2, early_stopping=True)

        # Decode the generated output
        output_text = tokenizer_t5.decode(outputs[0], skip_special_tokens=True)

        return output_text
    except Exception as e:
        return f"Error in generating text: {str(e)}"

# Example usage
input_text = "Apa ibu kota Indonesia?"
output_text = generate_text(input_text)
print(f"Input: {input_text}")
print(f"Output: {output_text}")
