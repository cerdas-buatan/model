import numpy as np
from tensorflow.keras.layers import Embedding, LSTM, Dense, Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
import os
import matplotlib.pyplot as plt

# Disable oneDNN optimizations warning
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

class Seq2Seq:
    def __init__(self, xseq_len, yseq_len, vocab_size, emb_dim, num_layers):
        self.xseq_len = xseq_len

        self.build_model()

class Seq2Seq:
    def __init__(self, xseq_len, yseq_len, vocab_size, emb_dim, num_layers):
        self.xseq_len = xseq_len

        self.build_model()

        
#


#
    def build_model(self):

        # Encoder
        encoder_inputs = Input(shape=(self.xseq_len,), name='encoder_inputs')
        enc_emb = Embedding(self.vocab_size, self.emb_dim, mask_zero=True)(encoder_inputs)
        encoder_outputs, state_h, state_c = LSTM(self.emb_dim, return_state=True, name='encoder_lstm')(enc_emb)
        encoder_states = [state_h, state_c]
        encoder_states = [state_h, state_c]

        # Decoder
        decoder_inputs = Input(shape=(self.yseq_len,), name='decoder_inputs')
        dec_emb = Embedding(self.vocab_size, self.emb_dim, mask_zero=True)(decoder_inputs)
        decoder_lstm = LSTM(self.emb_dim, return_sequences=True, return_state=True, name='decoder_lstm')
        decoder_outputs, _, _ = decoder_lstm(dec_emb, initial_state=encoder_states)
        decoder_outputs = Dense(self.vocab_size, activation='softmax', name='decoder_dense')(decoder_outputs)

        # Decoder
        decoder_inputs = Input(shape=(self.yseq_len,), name='decoder_inputs')
        dec_emb = Embedding(self.vocab_size, self.emb_dim, mask_zero=True)(decoder_inputs)
        decoder_lstm = LSTM(self.emb_dim, return_sequences=True, return_state=True, name='decoder_lstm')
        decoder_outputs, _, _ = decoder_lstm(dec_emb, initial_state=encoder_states)
        decoder_outputs = Dense(self.vocab_size, activation='softmax', name='decoder_dense')(decoder_outputs)

        # Define and compile the model
#
        # Define and compile the model
        self.model = Model([encoder_inputs, decoder_inputs], decoder_outputs)
        self.model.compile(optimizer=Adam(), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
#       self.model.compile(optimizer=Adam(), loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    def train(self, X_train, y_train, batch_size, epochs):
        y_train = np.expand_dims(y_train, axis=-1)
        history = self.model.fit([X_train, y_train[:, :-1]], y_train[:, 1:], batch_size=batch_size, epochs=epochs, validation_split=0.1)
        return history

    def evaluate(self, X_test, y_test):
        y_test = np.expand_dims(y_test, axis=-1)
        results = self.model.evaluate([X_test, y_test[:, :-1]], y_test[:, 1:])
        print(f'Test Loss: {results[0]}')
        print(f'Test Accuracy: {results[1]}')
        return results

#    def evaluate(self, X_test, y_test):

    def evaluate(self, X_test, y_test):
        y_test = np.expand_dims(y_test, axis=-1)
        results = self.model.evaluate([X_test, y_test[:, :-1]], y_test[:, 1:])
        print(f'Test Loss: {results[0]}')
        print(f'Test Accuracy: {results[1]}')
        return results
    

#        y_test = np.expand_dims(y_test, axis=-1)
#        return results
    
def preprocess_data(questions, answers, xseq_len, yseq_len, num_words):
    tokenizer = Tokenizer(num_words=num_words, oov_token='<OOV>')
    tokenizer.fit_on_texts(np.concatenate((questions, answers)))
    X = pad_sequences(tokenizer.texts_to_sequences(questions), maxlen=xseq_len, padding='post')
    y = pad_sequences(tokenizer.texts_to_sequences(answers), maxlen=yseq_len, padding='post')
    return X, y, tokenizer

#    tokenizer.fit_on_texts(np.concatenate((questions, answers)))

def preprocess_data(questions, answers, xseq_len, yseq_len, num_words):
    tokenizer = Tokenizer(num_words=num_words, oov_token='<OOV>')
    tokenizer.fit_on_texts(np.concatenate((questions, answers)))
    X = pad_sequences(tokenizer.texts_to_sequences(questions), maxlen=xseq_len, padding='post')
    y = pad_sequences(tokenizer.texts_to_sequences(answers), maxlen=yseq_len, padding='post')
    return X, y, tokenizer

# Load and preprocess the dataset
data = pd.read_csv('dataset_clean2.csv')
data['question'] = data['question'].astype(str).fillna('')
data['answer'] = data['answer'].astype(str).fillna('')
questions, answers = data['question'].values, data['answer'].values

#data = pd.read_csv('dataset_clean2.csv')


# Preprocess the data
X, y, tokenizer = preprocess_data(questions, answers, xseq_len, yseq_len, num_words)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Instantiate and train the model
model = Seq2Seq(xseq_len=xseq_len, yseq_len=yseq_len, vocab_size=num_words, emb_dim=emb_dim, num_layers=num_layers)
history = model.train(X_train, y_train, batch_size=16, epochs=3)

# Evaluate the model
results = model.evaluate(X_test, y_test)

# Evaluate the model
# Evaluate the model
results = model.evaluate(X_test, y_test)

# Plot training & validation accuracy values
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')
plt.show()
