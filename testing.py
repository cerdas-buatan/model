import tensorflow as tf
from transformers import TFT5ForConditionalGeneration, T5Tokenizer
import re

# Load the trained model and tokenizer
model_path = 't5_text_to_text_model'
model = TFT5ForConditionalGeneration.from_pretrained(model_path)
tokenizer = T5Tokenizer.from_pretrained(model_path)

# Function to clean and preprocess input text
def preprocess_text(text):
    # Remove repetitive characters
    text = re.sub(r'(.)\1{2,}', r'\1', text)
    return text

# Function to generate text from input
def generate_text(input_text):
    # Preprocess the input text
    input_text = preprocess_text(input_text)

    # Tokenize the input text
    input_ids = tokenizer.encode(input_text, return_tensors='tf', add_special_tokens=True, max_length=64, padding='max_length', truncation=True)

    # Generate text from the model
    outputs = model.generate(input_ids, max_length=64, num_beams=4, early_stopping=True)

    # Decode the generated output
    output_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

    return output_text

# Testing loop
while True:
    input_text = input("Masukkan pertanyaan Anda (atau ketik 'exit' untuk keluar): ")

    if input_text.lower() == 'exit':
        break

    # Generate text based on input
    generated_text = generate_text(input_text)
    print("Jawaban dari model:")
    print(generated_text)
