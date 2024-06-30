import tensorflow as tf
from transformers import TFBertForSequenceClassification, BertTokenizer
import pandas as pd
import re
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory

# Load the dataset
df = pd.read_csv('dataset-a-NewBackup', sep='|', usecols=['question', 'answer'])

# Preprocessing functions
factory = StemmerFactory()
stemmer = factory.create_stemmer()

punct_re_escape = re.compile('[%s]' % re.escape('!"#$%&()*+,./:;<=>?@[\\]^_`{|}~'))
unknowns = ["gak paham", "kurang ngerti", "I don't know", "I don't care"]

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
            normal_sentence += root_sentence+" "
        return punct_re_escape.sub('',normal_sentence)
    return sentence

# Clean and preprocess the dataset
filename= './dataset/clean_qa.txt'
with open(filename, 'w', encoding='utf-8') as f:
    for index, row in df.iterrows():
        question = normalize_sentence(str(row['question']))
        question = normalize_sentence(question)
        question = stemmer.stem(question)

        answer = str(row['answer']).lower().replace('gaysdisal', 'aku').replace('\n', ' ')

        if len(question.split()) > 0 and len(question.split()) < 13 and len(answer.split()) < 29:
            body = "{" + question + "}|<START> {" + answer + "} <END>"
            print(body, file=f)

# Load IndoBERT model and tokenizer
tokenizer = BertTokenizer.from_pretrained('indobenchmark/indobert-base-p2')
model = TFBertForSequenceClassification.from_pretrained('indobenchmark/indobert-base-p2')

# Example of using tokenizer
text = "contoh kalimat untuk di-tokenisasi"
encoded_input = tokenizer(text, return_tensors='tf')

# Example of training or fine-tuning the model
# Replace this with your actual training code based on the preprocessed data
# Example:
# model.compile(...)
# model.fit(...)

# Save the model and tokenizer
model.save_pretrained('./indobert_model')
tokenizer.save_pretrained('./indobert_model')





# import pandas as pd
# import re
# from transformers import TFBertForSequenceClassification, BertTokenizer
# import tensorflow as tf
# from sklearn.model_selection import train_test_split

# # Load the dataset
# df = pd.read_csv('dataset-a-NewBackup.csv')

# # Clean and combine the dataset
# def clean_text(text):
#     text = re.sub(r'[^\w\s]', '', text)  # Remove punctuation
#     text = re.sub(r'\s+', ' ', text).strip()  # Remove extra spaces
#     return text

# df['question'] = df['answer'].apply(clean_text)
# df['question'] = df['answer'].apply(clean_text)
# df['question|answer'] = df['question'] + '|' + df['answer']

# # Prepare the dataset
# tokenizer = BertTokenizer.from_pretrained('indobenchmark/indobert-base-p2')
# input_ids = []
# attention_masks = []
# labels = []

# for idx, row in df.iterrows():
#     pertanyaan, answer = row['question'], row['answer']
#     encoded = tokenizer.encode_plus(
#         pertanyaan, add_special_tokens=True, max_length=64, 
#         padding='max_length', return_attention_mask=True, truncation=True
#     )
#     input_ids.append(encoded['input_ids'])
#     attention_masks.append(encoded['attention_mask'])
#     labels.append(answer)  # Assuming answers are already in a numeric format

# input_ids = tf.constant(input_ids)
# attention_masks = tf.constant(attention_masks)
# labels = tf.constant(labels)

# # Split the dataset into training and testing sets
# train_input_ids, test_input_ids, train_attention_masks, test_attention_masks, train_labels, test_labels = train_test_split(
#     input_ids, attention_masks, labels, test_size=0.2, random_state=42)

# # Load model
# model = TFBertForSequenceClassification.from_pretrained('indobenchmark/indobert-base-p2', num_labels=len(set(labels.numpy())))

# # Compile the model
# model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=2e-5), 
#               loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), 
#               metrics=['accuracy'])

# # Train the model
# history = model.fit(
#     [train_input_ids, train_attention_masks], train_labels, 
#     validation_data=([test_input_ids, test_attention_masks], test_labels), 
#     epochs=3, batch_size=32
# )

# # Save the model
# model.save_pretrained('indobert_model')
# tokenizer.save_pretrained('indobert_model')

# # Evaluate the model
# loss, accuracy = model.evaluate([test_input_ids, test_attention_masks], test_labels)
# print(f'Test Loss: {loss}')
# print(f'Test Accuracy: {accuracy}')
