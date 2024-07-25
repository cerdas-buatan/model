import tensorflow as tf
import joblib
import os

# Define folder to load model and other files
save_dir = 'saved_model'

# Load model, vectorizer, and label encoder
model = tf.keras.models.load_model(os.path.join(save_dir, 'nn_model.h5'))
vectorizer = joblib.load(os.path.join(save_dir, 'vectorizer.pkl'))
label_encoder = joblib.load(os.path.join(save_dir, 'label_encoder.pkl'))

# Predict function
def generate_answer(question):
    # Transform question to BoW
    question_bow = vectorizer.transform([question]).toarray()
    # Predict with the model
    prediction = model.predict(question_bow)
    predicted_label = tf.argmax(prediction, axis=1).numpy()[0]
    # Decode label to original text
    return label_encoder.inverse_transform([predicted_label])[0]

# Manual input
while True:
    user_input = input("Enter a question (or type 'exit' to quit): ")
    if user_input.lower() == 'exit':
        break
    predicted_answer = generate_answer(user_input)
    print(f'Predicted answer: {predicted_answer}')