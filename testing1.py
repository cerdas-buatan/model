import tensorflow as tf
import joblib
import os

# Define folder to load model and other files
save_dir = 'saved_model'

# Load model, vectorizer, and label encoder
model = tf.keras.models.load_model(os.path.join(save_dir, 'nn_model.h5'))
vectorizer = joblib.load(os.path.join(save_dir, 'vectorizer.pkl'))
label_encoder = joblib.load(os.path.join(save_dir, 'label_encoder.pkl'))

