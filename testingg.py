import tensorflow as tf
from transformers import TFT5ForConditionalGeneration, T5Tokenizer

# Load the trained model and tokenizer
model_path = 't5_text_to_text_model'

try:
    model = TFT5ForConditionalGeneration.from_pretrained(model_path)
    tokenizer = T5Tokenizer.from_pretrained(model_path)
except Exception as e:
    print(f"Error loading model or tokenizer: {str(e)}")
    exit()

# Function to generate text from input
def generate_text(input_text):
    try:
        # Tokenize the input text
        inputs = tokenizer.encode_plus(input_text, return_tensors='tf', add_special_tokens=True, max_length=50, padding='max_length', truncation=True)
        input_ids = inputs['input_ids']
        attention_mask = inputs['attention_mask']

        # Generate text from the model
        outputs = model.generate(input_ids, attention_mask=attention_mask, max_length=50, num_beams=2, early_stopping=True)

        # Decode the generated output
        output_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

        return output_text
    except Exception as e:
        return f"Error in generating text: {str(e)}"

# Example usage
input_text = "Apa ibu kota Indonesia?"
output_text = generate_text(input_text)
print(f"Input: {input_text}")
print(f"Output: {output_text}")
