import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.layers import Embedding, LSTM, Dense, Input # type: ignore # type: ignore
from tensorflow.keras.models import Model # type: ignore 
from tensorflow.keras.optimizers import Adam # type: ignore
import os
import matplotlib.pyplot as plt

# Disable oneDNN optimizations warning
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

class Seq2Seq:
    def __init__(self, xseq_len, yseq_len, xvocab_size, yvocab_size, emb_dim, num_layers):  # Perbaikan disini
        self.xseq_len = xseq_len
        self.yseq_len = yseq_len
        self.xvocab_size = xvocab_size
        self.yvocab_size = yvocab_size
        self.emb_dim = emb_dim
        self.num_layers = num_layers
        self.build_model()

    def build_model(self):
        # Encoder
        encoder_inputs = Input(shape=(self.xseq_len,), name='encoder_inputs')
        enc_emb = Embedding(self.xvocab_size, self.emb_dim, mask_zero=True)(encoder_inputs)
        
        encoder_lstm = LSTM(self.emb_dim, return_state=True, name='encoder_lstm')
        encoder_outputs, state_h, state_c = encoder_lstm(enc_emb)
        encoder_states = [state_h, state_c]

        # Decoder
        decoder_inputs = Input(shape=(self.yseq_len,), name='decoder_inputs')
        dec_emb = Embedding(self.yvocab_size, self.emb_dim, mask_zero=True)(decoder_inputs)

        decoder_lstm = LSTM(self.emb_dim, return_sequences=True, return_state=True, name='decoder_lstm')
        decoder_outputs, _, _ = decoder_lstm(dec_emb, initial_state=encoder_states)

        decoder_dense = Dense(self.yvocab_size, activation='softmax', name='decoder_dense')
        decoder_outputs = decoder_dense(decoder_outputs)

        # Define the model
        self.model = Model([encoder_inputs, decoder_inputs], decoder_outputs)
        
        # Compile the model with sparse categorical crossentropy loss and accuracy metric
        self.model.compile(optimizer=Adam(), loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    def train(self, X_train, y_train, batch_size, epochs):
        # Ensure y_train is in the right shape
        y_train = np.expand_dims(y_train, axis=-1)
        
        # Train the model
        history = self.model.fit([X_train, y_train[:, :]], y_train[:, :], batch_size=batch_size, epochs=epochs, validation_split=0.1)
        return history
        #history = self.model.fit([X_train, y_train[:, :]], y_train[:, :], batch_size=batch_size, epochs=epochs, validation_split=0.1)
        #return history

    def evaluate(self, X_test, y_test):
        # Ensure y_test is in the right shape
        y_test = np.expand_dims(y_test, axis=-1)
        
        # Evaluate the model
        results = self.model.evaluate([X_test, y_test[:, :]], y_test[:, :])
        print(f'Test Loss: {results[0]}')
        print(f'Test Accuracy: {results[1]}')

    def add_data(self, X_new, y_new):
        global X_train, y_train
        X_train = np.vstack((X_train, X_new))
        y_train = np.vstack((y_train, y_new))

# Load and preprocess the dataset
data = pd.read_csv('dataset_clean2.csv')

# Ensure all data in 'question' and 'answer' columns are strings and handle missing values
data['question'] = data['question'].astype(str).fillna('')
data['answer'] = data['answer'].astype(str).fillna('')

questions = data['question'].values
answers = data['answer'].values

# Define a function to tokenize and pad sequences
from tensorflow.keras.preprocessing.text import Tokenizer # type: ignore
from tensorflow.keras.preprocessing.sequence import pad_sequences # type: ignore

def preprocess_data(questions, answers, xseq_len, yseq_len, num_words):
    tokenizer = Tokenizer(num_words=num_words, oov_token='<OOV>')
    tokenizer.fit_on_texts(np.concatenate((questions, answers)))
    
    X = tokenizer.texts_to_sequences(questions)
    X = pad_sequences(X, maxlen=xseq_len, padding='post')
    
    y = tokenizer.texts_to_sequences(answers)
    y = pad_sequences(y, maxlen=yseq_len, padding='post')
    
    return X, y, tokenizer

# Set the parameters
xseq_len = 25
yseq_len = 25
num_words = 1500
emb_dim = 130
num_layers = 3

# Preprocess the data
X, y, tokenizer = preprocess_data(questions, answers, xseq_len, yseq_len, num_words)

# Split the data into training and testing sets
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Instantiate and train the model
model = Seq2Seq(xseq_len=xseq_len, yseq_len=yseq_len, xvocab_size=num_words, yvocab_size=num_words, emb_dim=emb_dim, num_layers=num_layers)
history = model.train(X_train, y_train, batch_size=16, epochs=3)

# Evaluate the model
model.evaluate(X_test, y_test)

# Display training history
# Plot training & validation accuracy values
#plt.plot(history.history['accuracy'])
#plt.plot(history.history['val_accuracy'])
#plt.title('Model accuracy')  
#plt.ylabel('Accuracy')  
#plt.xlabel('Epoch')  
#plt.legend(['Train', 'Validation'], loc='upper left')  
#plt.show()  
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')
plt.show()

# Plot training & validation loss values
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')
plt.show()