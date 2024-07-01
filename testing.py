import tensorflow as tf
from transformers import TFBertForSequenceClassification, BertTokenizer
import numpy as np

# Load the model and tokenizer
model_path = 'indobert_model'
model = TFBertForSequenceClassification.from_pretrained(model_path)
tokenizer = BertTokenizer.from_pretrained(model_path)

def predict_answer(question):
    # Tokenize the input question
    encoded = tokenizer.encode_plus(question, add_special_tokens=True, max_length=64, padding='max_length', return_attention_mask=True, truncation=True)
    input_ids = tf.constant([encoded['input_ids']])
    attention_mask = tf.constant([encoded['attention_mask']])
    
    # Make prediction
    output = model([input_ids, attention_mask])
    logits = output.logits
    predicted_class = np.argmax(logits, axis=1).numpy()[0]
    
    return predicted_class

# Loop to continuously take input from the user
while True:
    sample_question = input("Enter a question (or type 'exit' to stop): ")
    if sample_question.lower() == 'exit':
        break
    predicted_answer = predict_answer(sample_question)
    print(f"Predicted answer for '{sample_question}': {predicted_answer}")
