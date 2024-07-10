import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.layers import Embedding, LSTM, Dense, Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
import os
import matplotlib.pyplot as plt

# Disable oneDNN optimizations warning
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

class Seq2Seq:
    def _init_(self, xseq_len, yseq_len, xvocab_size, yvocab_size, emb_dim, num_layers):
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
data = pd.read_csv('dataset_clean2.csv', sep='|')

# Ensure all data in 'question' and 'answer' columns are strings and handle missing values
data['question'] = data['question'].astype(str).fillna('')
data['answer'] = data['answer'].astype(str).fillna('')

questions = data['question'].values
answers = data['answer'].values


