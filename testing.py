import tensorflow as tf
from transformers import TFBertForSequenceClassification, BertTokenizer
import pandas as pd

# Load the saved model and tokenizer
model_path = 'indobert_model'
model = TFBertForSequenceClassification.from_pretrained(model_path)
tokenizer = BertTokenizer.from_pretrained(model_path)

# Define a function to generate answer based on input question
def generate_answer(question):
    # Tokenize the input question
    inputs = tokenizer(question, return_tensors="tf")
    
    # Make predictions
    outputs = model(**inputs)
    predicted_class = tf.argmax(outputs.logits, axis=1).numpy()[0]
    
    # Assuming labels are mapped to answers in some way
    predicted_answer = str(predicted_class)  # Example: Just an example, replace with actual logic
    
    return predicted_answer

# Testing with a new question
new_question = input("Masukkan pertanyaan Anda: ")
predicted_answer = generate_answer(new_question)
print(f"Predicted answer: {predicted_answer}")
