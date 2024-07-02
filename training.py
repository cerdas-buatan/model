import tensorflow as tf
from transformers import TFBertForSequenceClassification, BertTokenizer
import pandas as pd
import re
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
import nltk
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# nltk.download('stopwords')

# # Stopwords dalam bahasa Indonesia
# stop_words = set(stopwords.words('indonesian'))

# # Kamus kata slang dan singkatan
# slang_dict = {
#     "gak": "tidak",
#     "nggak": "tidak",
#     "ga": "tidak",
#     "enggak": "tidak",
#     "tdk": "tidak",
#     "gini": "begini",
#     "gitu": "begitu",
#     "trs": "terus",
#     "tp": "tapi",
#     "sbnr": "sebenarnya",
#     "sm": "sama",
#     "sbg": "sebagai",
#     "blg": "bilang",
#     "krn": "karena",
#     "jg": "juga",
#     "aj": "saja",
#     "udh": "sudah",
#     "jd": "jadi",
#     "kyk": "seperti",
#     "dlm": "dalam"
#     # Tambahkan lebih banyak kata sesuai kebutuhan
# }

# Load the dataset
df = pd.read_csv('dataset_clean.csv', sep='|', usecols=['question', 'answer'])

# Preprocessing functions
factory = StemmerFactory()
stemmer = factory.create_stemmer()

punct_re_escape = re.compile('[%s]' % re.escape('!"#$%&()*+,./:;<=>?@[\\]^_`{|}~'))
unknowns = ["gak paham", "kurang ngerti", "I don't know", "I don't care"]

# def check_normal_word(word):
#     # Mengganti kata slang dengan bentuk formal
#     if word in slang_dict:
#         word = slang_dict[word]
    
#     # Menghapus stopwords
#     if word in stop_words:
#         return ""
    
#     return word

def normalize_sentence(sentence):
    sentence = punct_re_escape.sub('', sentence.lower())
    sentence = sentence.replace('gaysdisal', '').replace('\n', '').replace(' njir','').replace('njinx','').replace(' dong','').replace(' sih','').replace(' deh','').replace(' duh','').replace(' ea','').replace(' aw','')
    sentence = sentence.replace('gaysdisal', '')
    sentence = re.sub(r'((wk)+(w?)+(k?)+)+', '', sentence)
    sentence = re.sub(r'((xi)+(x?)+(i?)+)+', '', sentence)
    sentence = re.sub(r'((h(a|i|e)h)((a|i|e)?)+(h?)+((a|i|e)?)+)+', '', sentence)
    sentence = ' '.join(sentence.split())
    if sentence:
        sentence = sentence.strip().split(" ")
        normal_sentence = " "
        for word in sentence:
            normalize_word = check_normal_word(word)
            root_sentence = stemmer.stem(normalize_word)
            normal_sentence += root_sentence + " "
        return punct_re_escape.sub('', normal_sentence.strip())
    return sentence

# Clean and preprocess the dataset
cleaned_data = []
for index, row in df.iterrows():
    question = normalize_sentence(str(row['question']))
    answer = str(row['answer']).lower().replace('gaysdisal', 'aku').replace('\n', ' ')

    if len(question.split()) > 0 and len(question.split()) < 13 and len(answer.split()) < 29:
        cleaned_data.append({"question": question, "answer": answer})

df_cleaned = pd.DataFrame(cleaned_data)

# Tokenisasi dan persiapan data
tokenizer = BertTokenizer.from_pretrained('indobenchmark/indobert-base-p2')
input_ids = []
attention_masks = []

for index, row in df_cleaned.iterrows():
    encoded = tokenizer.encode_plus(
        row['question'],
        add_special_tokens=True,
        max_length=64,
        pad_to_max_length=True,
        return_attention_mask=True,
        return_tensors='tf'
    )
    input_ids.append(encoded['input_ids'])
    attention_masks.append(encoded['attention_mask'])

input_ids = tf.concat(input_ids, axis=0)
attention_masks = tf.concat(attention_masks, axis=0)

# Split data menjadi train dan test set
train_inputs, validation_inputs, train_labels, validation_labels = train_test_split(input_ids, attention_masks, test_size=0.1)

# Konversi ke TensorFlow Dataset
train_dataset = tf.data.Dataset.from_tensor_slices(({'input_ids': train_inputs, 'attention_mask': train_labels}, train_labels))
validation_dataset = tf.data.Dataset.from_tensor_slices(({'input_ids': validation_inputs, 'attention_mask': validation_labels}, validation_labels))

# Load model
model = TFBertForSequenceClassification.from_pretrained('indobenchmark/indobert-base-p2', num_labels=2)

# Compile model
optimizer = tf.keras.optimizers.Adam(learning_rate=3e-5)
loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
model.compile(optimizer=optimizer, loss=loss, metrics=['accuracy'])

# Train model
epochs = 3
batch_size = 16

history = model.fit(train_dataset.shuffle(100).batch(batch_size), epochs=epochs, batch_size=batch_size, validation_data=validation_dataset.batch(batch_size))

# Save the model
model.save_pretrained('./indobert_model')
tokenizer.save_pretrained('./indobert_model')

# Evaluate the model
results = model.evaluate(validation_dataset.batch(batch_size))
print(f"Validation Loss: {results[0]}")
print(f"Validation Accuracy: {results[1]}")

# Function to predict answer
def predict_answer(question):
    question = normalize_sentence(question)
    inputs = tokenizer(question, return_tensors="tf", max_length=64, padding="max_length", truncation=True)
    outputs = model(inputs)
    logits = outputs.logits
    predicted_label = tf.argmax(logits, axis=-1).numpy()
    return predicted_label

# Input pertanyaan dan perhitungan akurasi
correct_answers = 0
total_questions = 0

for index, row in df_cleaned.iterrows():
    predicted_answer = predict_answer(row['question'])
    actual_answer = row['answer']
    total_questions += 1
    if predicted_answer == actual_answer:
        correct_answers += 1

accuracy = correct_answers / total_questions
print(f"Model Accuracy: {accuracy}")
