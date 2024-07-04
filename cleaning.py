import tensorflow as tf
from transformers import TFBertForSequenceClassification, BertTokenizer
import pandas as pd
import re
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
import nltk
from nltk.corpus import stopwords
nltk.download('stopwords')

# Stopwords dalam bahasa Indonesia
stop_words = set(stopwords.words('indonesian'))

# Kamus kata slang dan singkatan
slang_dict = {
    "gak": "tidak",
    "nggak": "tidak",
    "ga": "tidak",
    "enggak": "tidak",
    "tdk": "tidak",
    "gini": "begini",
    "gitu": "begitu",
    "trs": "terus",
    "tp": "tapi",
    "sbnr": "sebenarnya",
    "sm": "sama",
    "sbg": "sebagai",
    "blg": "bilang",
    "krn": "karena",
    "jg": "juga",
    "aj": "saja",
    "udh": "sudah",
    "jd": "jadi",
    "kyk": "seperti",
    "dlm": "dalam"
    # Tambahkan lebih banyak kata sesuai kebutuhan
}

# Load the dataset
df = pd.read_csv('dataset_clean.csv', sep='|', usecols=['question', 'answer'])

# Preprocessing functions
factory = StemmerFactory()
stemmer = factory.create_stemmer()

punct_re_escape = re.compile('[%s]' % re.escape('!"#$%&()*+,./:;<=>?@[\\]^_`{|}~'))
unknowns = ["gak paham", "kurang ngerti", "I don't know", "I don't care"]

def check_normal_word(word):
    # Mengganti kata slang dengan bentuk formal
    if word in slang_dict:
        word = slang_dict[word]
    
    # Menghapus stopwords
    if word in stop_words:
        return ""
    
    return word

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


def check_normal_word(word):
    # Tambahkan logika normalisasi tambahan jika diperlukan
    return word

# Clean and preprocess the dataset
cleaned_data = []
for index, row in df.iterrows():
    question = normalize_sentence(str(row['question']))
    answer = str(row['answer']).lower().replace('gaysdisal', 'aku').replace('\n', ' ')

    if len(question.split()) > 0 and len(question.split()) < 13 and len(answer.split()) < 29:
        body = "" + question + "|" + answer + ""
        cleaned_data.append(body)

# Save cleaned data to a file
filename = './dataset/clean_qa.txt'
with open(filename, 'w', encoding='utf-8') as f:
    for item in cleaned_data:
        f.write("%s\n" % item)

# Load IndoBERT model and tokenizer
tokenizer = BertTokenizer.from_pretrained('indobenchmark/indobert-base-p2')
model = TFBertForSequenceClassification.from_pretrained('indobenchmark/indobert-base-p2')

# Example of using tokenizer
# text = "contoh kalimat untuk di-tokenisasi"
# encoded_input = tokenizer(text, return_tensors='tf')

# Save the model and tokenizer
model.save_pretrained('./indobert_model')
tokenizer.save_pretrained('./indobert_model')

