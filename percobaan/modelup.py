from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
import re
import pandas as pd

factory = StemmerFactory()
stemmer = factory.create_stemmer()

punct_re_escape = re.compile('[%s]' % re.escape('!"#$%&()*+,./:;<=>?@[\\]^_`{|}~'))

def replace_symbols(text, replace_dict):
    for symbol, replacement in replace_dict.items():
        text = text.replace(symbol, replacement)
    return text

def normalize_sentence(sentence):
    sentence = punct_re_escape.sub('', sentence.lower())
    sentence = sentence.replace('iteung', '').replace('\n', '').replace(' wah','').replace('wow','').replace(' dong','').replace(' sih','').replace(' deh','')
    sentence = sentence.replace('teung', '')
    sentence = re.sub(r'((wk)+(w?)+(k?)+)+', '', sentence)
    sentence = re.sub(r'((xi)+(x?)+(i?)+)+', '', sentence)
    sentence = re.sub(r'((h(a|i|e)h)((a|i|e)?)+(h?)+((a|i|e)?)+)+', '', sentence)
    sentence = ' '.join(sentence.split())
    return sentence

def preprocess_text(text):
    text = normalize_sentence(text)
    # Additional processing steps if needed
    text = ' '.join(stemmer.stem(word) for word in text.split())
    return text

# Load dataset
df = pd.read_csv('data.csv', delimiter="|", on_bad_lines='skip')

# Clean and preprocess data
df['question'] = df['question'].apply(preprocess_text)
df['answer'] = df['answer'].apply(preprocess_text)

df.to_csv('dataset_cleaned.csv', index=False)
print('Cleaned dataset saved as dataset_cleaned.csv')
