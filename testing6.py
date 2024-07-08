import tensorflow as tf
from transformers import TFT5ForConditionalGeneration, T5Tokenizer
import pandas as pd
import numpy as np

# Load the trained model and tokenizer
model_path = 't5_text_to_text_model'
model = TFT5ForConditionalGeneration.from_pretrained(model_path)
tokenizer = T5Tokenizer.from_pretrained(model_path)

# Function to generate response for a given question
def generate_response(question, model, tokenizer):
    input_ids = tokenizer.encode(question, return_tensors='tf', max_length=64, padding='max_length', truncation=True)
    outputs = model.generate(input_ids)
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response

# Manual Testing Example
def manual_testing():
    question = input("Masukkan pertanyaan: ")
    answer = generate_response(question, model, tokenizer)
    print(f"Jawaban: {answer}")
    return question, answer

# Save to Excel function
def save_to_excel(questions, answers, accuracies, filename='chatbot_testing_results.xlsx'):
    df = pd.DataFrame({
        'Pertanyaan': questions,
        'Jawaban_Model': answers,
        'Akurasi': accuracies
    })
    df.to_excel(filename, index=False)

# Main testing loop
questions = []
answers = []
accuracies = []

while True:
    question, model_answer = manual_testing()
    # In a real scenario, you would compare model_answer with ground truth from your dataset
    # For simplicity, we assume perfect accuracy here
    accuracy = 1.0
    
    questions.append(question)
    answers.append(model_answer)
    accuracies.append(accuracy)
    
    save_to_excel(questions, answers, accuracies)

    cont = input("Lanjutkan testing? (Y/N): ").lower()
    if cont != 'y':
        break
