import tensorflow as tf
from transformers import TFT5ForConditionalGeneration, T5Tokenizer, AutoTokenizer, TFAutoModelForSequenceClassification
import pandas as pd
import csv
from sklearn.preprocessing import LabelEncoder

def load_model_and_tokenizer(model_path, bert_model_name):
    """
    Load the T5 model and tokenizer, and the IndoBERT tokenizer and model.
    """
    try:
        model_t5 = TFT5ForConditionalGeneration.from_pretrained(model_path)
        tokenizer_t5 = T5Tokenizer.from_pretrained(model_path)
    except Exception as e:
        raise RuntimeError(f"Error loading T5 model or tokenizer: {str(e)}")

    try:
        tokenizer_bert = AutoTokenizer.from_pretrained(bert_model_name)
        model_bert = TFAutoModelForSequenceClassification.from_pretrained(bert_model_name, from_pt=True)
    except Exception as e:
        raise RuntimeError(f"Error loading IndoBERT model or tokenizer: {str(e)}")
    
    return model_t5, tokenizer_t5, model_bert, tokenizer_bert

def load_and_preprocess_data(filepath):
    """
    Load and preprocess the dataset.
    """
    with open(filepath, 'r', encoding='utf-8') as file:
        reader = csv.reader(file, delimiter='|')
        filtered_rows = [row for row in reader if len(row) == 2 and row[0].strip() != "" and row[1].strip() != ""]

    df = pd.DataFrame(filtered_rows, columns=['question', 'answer'])
    label_encoder = LabelEncoder()
    label_encoder.fit(df['answer'])
    
    return df, label_encoder

def generate_text(input_text, tokenizer, model):
    """
    Generate text from the input using the T5 model.
    """
    try:
        inputs = tokenizer.encode_plus(input_text, return_tensors='tf', add_special_tokens=True, max_length=50, padding='max_length', truncation=True)
        input_ids = inputs['input_ids']
        attention_mask = inputs['attention_mask']
        
        outputs = model.generate(input_ids, attention_mask=attention_mask, max_length=50, num_beams=2, early_stopping=True)
        output_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

        return output_text
    except Exception as e:
        return f"Error in generating text: {str(e)}"

def main():
    model_path = 't5_text_to_text_model'
    bert_model_name = 'indolem/indobert-base-uncased'
    dataset_path = 'dataset_clean2.csv'

    # Load models and tokenizers
    model_t5, tokenizer_t5, model_bert, tokenizer_bert = load_model_and_tokenizer(model_path, bert_model_name)

    # Load and preprocess data
    df, label_encoder = load_and_preprocess_data(dataset_path)

    # Example usage
    input_text = "Apa ibu kota Indonesia?"
    output_text = generate_text(input_text, tokenizer_t5, model_t5)
    
    print(f"Input: {input_text}")
    print(f"Output: {output_text}")

if __name__ == "__main__":
    main()
