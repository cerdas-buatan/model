import tensorflow as tf
from transformers import TFT5ForConditionalGeneration, T5Tokenizer, AutoTokenizer, TFAutoModelForSequenceClassification
import pandas as pd
import csv
from sklearn.preprocessing import LabelEncoder
from pathlib import Path
import logging

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
def load_dataset(file_path: str):
    """Load and clean the dataset."""
    with open(file_path, 'r', encoding='utf-8') as file:
        reader = csv.reader(file, delimiter='|')
        filtered_rows = [row for row in reader if len(row) == 2 and row[0].strip() != "" and row[1].strip() != ""]
    return pd.DataFrame(filtered_rows, columns=['question', 'answer'])

# Function to generate text from input
def generate_text(self, input_text: str) -> str:
        """Generate text from input using T5 model."""
        try:
            inputs = self.tokenizer_t5.encode_plus(input_text, return_tensors='tf', add_special_tokens=True, max_length=50, padding='max_length', truncation=True)
            input_ids = inputs['input_ids']
            attention_mask = inputs['attention_mask']

            outputs = self.model_t5.generate(input_ids, attention_mask=attention_mask, max_length=50, num_beams=2, early_stopping=True)

            output_text = self.tokenizer_t5.decode(outputs[0], skip_special_tokens=True)
            return output_text
        except Exception as e:
            return f"Error in generating text: {str(e)}"

# Example usage
input_text = "Apa ibu kota Indonesia?"
output_text = generate_text(input_text)
print(f"Input: {input_text}")
print(f"Output: {output_text}")

logging.basicConfig(level=logging.ERROR)

def load_model_and_tokenizer(model_path: str, model_class, tokenizer_class):
    """Load the specified model and tokenizer."""
    try:
        model = model_class.from_pretrained(model_path)
        tokenizer = tokenizer_class.from_pretrained(model_path)
        return model, tokenizer
    except Exception as e:
        logging.error(f"Error loading {model_class.__name__} or tokenizer: {str(e)}")
        exit()

class TextGenerator:
    def __init__(self, t5_model_path: str, bert_model_path: str, dataset_path: str):
        self.model_t5, self.tokenizer_t5 = load_model_and_tokenizer(t5_model_path, TFT5ForConditionalGeneration, T5Tokenizer)
        self.tokenizer_bert, self.model_bert = load_model_and_tokenizer(bert_model_path, TFAutoModelForSequenceClassification, AutoTokenizer)
        self.dataset = load_dataset(dataset_path)
        self.label_encoder = LabelEncoder()
        self.label_encoder.fit(self.dataset['answer'])

def main():
    t5_model_path = 't5_text_to_text_model'
    bert_model_path = 'indolem/indobert-base-uncased'
    dataset_path = 'dataset_clean2.csv'

    text_generator = TextGenerator(t5_model_path, bert_model_path, dataset_path)
    
    input_text = "Apa ibu kota Indonesia?"
    output_text = text_generator.generate_text(input_text)

if __name__ == "__main__":
    main()