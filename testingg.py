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

# Function to calculate accuracy
def calculate_accuracy(validation_data):
    true_answers = []
    predicted_answers = []

    for question, true_answer in validation_data:
        predicted_answer = generate_text(question)
        true_answers.append(true_answer)
        predicted_answers.append(predicted_answer)
        print(f"Question: {question}")
        print(f"Predicted Answer: {predicted_answer}")
        print(f"True Answer: {true_answer}")
        print("-" * 50)

        # Calculate accuracy using sklearn's accuracy_score
    accuracy = accuracy_score(true_answers, predicted_answers)
    return accuracys

# Example validation data
validation_data = [
    ("Apa ibu kota Indonesia?", "Jakarta"),
    ("Siapa presiden pertama Amerika Serikat?", "George Washington"),
    # Tambahkan lebih banyak data validasi di sini
]

# Example usage
if __name__ == "__main__":
    while True:
        input_text = input("Masukkan pertanyaan (atau ketik 'exit' untuk keluar): ")

        if input_text.lower() == 'exit':
            break

        # Generate text based on input
        generated_text = generate_text(input_text)
        print("Answer:")
        print(generated_text)
