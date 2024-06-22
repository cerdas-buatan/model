# import pandas as pd
# from transformers import AutoTokenizer

# # Load dataset
# df = pd.read_csv('dataset/a-backup.csv')
# texts = df['text'].tolist()

# # Initialize tokenizer
# tokenizer = AutoTokenizer.from_pretrained('microsoft/DialoGPT-medium')

# # Tokenize and encode data
# encodings = [tokenizer(text, return_tensors='pt') for text in texts]

import re
import pandas as pd

def preprocess_text(text):
    # Bersihkan teks dari karakter yang tidak diinginkan
    text = text.lower()
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
    return text

df = pd.read_csv('.dataset/a-backup.csv')
df['text'] = df['text'].apply(preprocess_text)
texts = df['text'].tolist()

from collections import Counter

def build_vocab(texts):
    words = []
    for text in texts:
        words.extend(text.split())
    word_counts = Counter(words)
    vocab = {word: idx for idx, (word, _) in enumerate(word_counts.items(), start=1)}
    vocab['<PAD>'] = 0  # Padding token
    return vocab

# Contoh penggunaan
vocab = build_vocab(texts)
def text_to_tokens(text, vocab):
    tokens = [vocab.get(word, vocab['<PAD>']) for word in text.split()]
    return tokens

# Contoh penggunaan
tokenized_texts = [text_to_tokens(text, vocab) for text in texts]
def tokenizer(text, vocab):
    tokens = text_to_tokens(text, vocab)
    return tokens

def detokenizer(tokens, vocab):
    reverse_vocab = {idx: word for word, idx in vocab.items()}
    words = [reverse_vocab.get(token, '<UNK>') for token in tokens]
    return ' '.join(words)

# 
# sample_text = "hello how are you"
tokens = tokenizer(sample_text, vocab)
print("Tokens:", tokens)

detokenized_text = detokenizer(tokens, vocab)
print("Detokenized Text:", detokenized_text)
