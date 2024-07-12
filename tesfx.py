import tensorflow as tf
from transformers import TFT5ForConditionalGeneration, T5Tokenizer

# Load the trained model and tokenizer
model_path = 't5_text_to_text_model'
model = TFT5ForConditionalGeneration.from_pretrained(model_path)
tokenizer = T5Tokenizer.from_pretrained(model_path)

# Function to generate text from input
def generate_text(input_text):
    try:
        # Tokenize the input text
        inputs = tokenizer.encode_plus(input_text, return_tensors='tf', add_special_tokens=True, max_length=64, padding='max_length', truncation=True)
        input_ids = inputs['input_ids']
        attention_mask = inputs['attention_mask']

        # Generate text from the model
        outputs = model.generate(input_ids, attention_mask=attention_mask, max_length=64, num_beams=4, early_stopping=True)

        # Decode the generated output
        output_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

        return output_text
    except Exception as e:
        return f"Error in generating text: {str(e)}"

# Example usage
while True:
    input_text = input("Masukkan pertanyaan Anda (atau ketik 'exit' untuk keluar): ")

    if input_text.lower() == 'exit':
        break

    # Generate text based on input
    generated_text = generate_text(input_text)
    print("Jawaban dari model:")
    print(generated_text)