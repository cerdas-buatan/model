import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import tensorflow as tf
import joblib
import os
import json
import uuid

# Define folder to save model and other files
save_dir = 'save_model2'
os.makedirs(save_dir, exist_ok=True)

# Initialize an empty list to store cleaned rows
rows = []

# Read and clean dataset, handling any anomalies
with open('data.csv', 'r', encoding='utf-8') as file:
    for line_number, line in enumerate(file):
        # Split line based on '|'
        parts = line.strip().split('|')
        if len(parts) == 2:  # Only process lines with exactly two parts
            rows.append(parts)

# Convert cleaned rows to DataFrame
df = pd.DataFrame(rows, columns=['question', 'answer'])

# Handle missing values
df.dropna(inplace=True)

# Remove duplicates
df.drop_duplicates(inplace=True)

# Initialize CountVectorizer
vectorizer = CountVectorizer()

# Fit and transform text to BoW
X_bow = vectorizer.fit_transform(df['question'])

# Use the 'answer' column as the label
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(df['answer'])

# Split dataset into training and testing sets (80% train, 20% test)
X_train_bow, X_test_bow, y_train, y_test = train_test_split(X_bow, y, test_size=0.2, random_state=42)

# Convert data to TensorFlow dataset
train_dataset = tf.data.Dataset.from_tensor_slices((X_train_bow.toarray(), y_train)).batch(128)
test_dataset = tf.data.Dataset.from_tensor_slices((X_test_bow.toarray(), y_test)).batch(128)

# Define Neural Network model
model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(X_train_bow.shape[1],)),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(len(label_encoder.classes_), activation='softmax')
])

# Compile model
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Train model
model.fit(train_dataset, epochs=64)

# Save model, vectorizer, and label encoder in the specified folder
model.save(os.path.join(save_dir, 'nn_model.h5'))
joblib.dump(vectorizer, os.path.join(save_dir, 'vectorizer.pkl'))
joblib.dump(label_encoder, os.path.join(save_dir, 'label_encoder.pkl'))

print(f"Training complete. Model, vectorizer, and label encoder saved in '{save_dir}'.")

# Predict on test data
predictions = model.predict(X_test_bow.toarray())
predicted_labels = label_encoder.inverse_transform(tf.argmax(predictions, axis=1).numpy())

# Create JSON output for MongoDB
output = []
for question, answer in zip(df['question'].iloc[X_test_bow.indices], predicted_labels):
    output.append({
        "_id": {"$oid": str(uuid.uuid4())},
        "question": question,
        "predicted_answer": answer
    })

# Save output to JSON file
with open('gaysdisal.json', 'w', encoding='utf-8') as f:
    json.dump(output, f, ensure_ascii=False, indent=4)

print(f"Output JSON saved to 'gaysdisal.json'")
