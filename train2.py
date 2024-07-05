import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Embedding, LSTM, Dense
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
import os

# Disable oneDNN optimizations warning
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

# Use the compatible version of sparse_softmax_cross_entropy
from tensorflow.compat.v1.losses import sparse_softmax_cross_entropy

class Seq2Seq:
    def __init__(self, xseq_len, yseq_len, xvocab_size, yvocab_size, emb_dim, num_layers):
        self.xseq_len = xseq_len
        self.yseq_len = yseq_len
        self.xvocab_size = xvocab_size
        self.yvocab_size = yvocab_size
        self.emb_dim = emb_dim
        self.num_layers = num_layers
        self.build_model()

    def build_model(self):
        # Encoder
        encoder_inputs = tf.keras.Input(shape=(self.xseq_len,), name='encoder_inputs')
        enc_emb = Embedding(self.xvocab_size, self.emb_dim, mask_zero=True)(encoder_inputs)
        
        encoder_lstm = LSTM(self.emb_dim, return_state=True, name='encoder_lstm')
        encoder_outputs, state_h, state_c = encoder_lstm(enc_emb)
        encoder_states = [state_h, state_c]

        # Decoder
        decoder_inputs = tf.keras.Input(shape=(self.yseq_len,), name='decoder_inputs')
        dec_emb = Embedding(self.yvocab_size, self.emb_dim, mask_zero=True)(decoder_inputs)

        decoder_lstm = LSTM(self.emb_dim, return_sequences=True, return_state=True, name='decoder_lstm')
        decoder_outputs, _, _ = decoder_lstm(dec_emb, initial_state=encoder_states)

        decoder_dense = Dense(self.yvocab_size, activation='softmax', name='decoder_dense')
        decoder_outputs = decoder_dense(decoder_outputs)

        # Define the model
        self.model = Model([encoder_inputs, decoder_inputs], decoder_outputs)
        
        # Compile the model with a custom accuracy metric
        self.model.compile(optimizer=Adam(), loss=sparse_softmax_cross_entropy, metrics=[self.accuracy])

    def accuracy(self, y_true, y_pred):
        # Custom accuracy function to compare the true and predicted values
        y_true = tf.argmax(y_true, axis=-1)
        y_pred = tf.argmax(y_pred, axis=-1)
        return tf.reduce_mean(tf.cast(tf.equal(y_true, y_pred), tf.float32))

    def train(self, X_train, y_train, batch_size, epochs):
        # Ensure y_train is in the right shape
        y_train = np.expand_dims(y_train, axis=-1)
        
        # Train the model
        self.model.fit([X_train, y_train[:, :]], y_train[:, :], batch_size=batch_size, epochs=epochs, validation_split=0.1)

    def evaluate(self, X_test, y_test):
        # Ensure y_test is in the right shape
        y_test = np.expand_dims(y_test, axis=-1)
        
        # Evaluate the model
        results = self.model.evaluate([X_test, y_test[:, :]], y_test[:, :])
        print(f'Test Loss: {results[0]}')
        print(f'Test Accuracy: {results[1]}')

# Example usage
X_train = np.random.randint(0, 1000, size=(100, 25))
y_train = np.random.randint(0, 1000, size=(100, 25))
X_test = np.random.randint(0, 1000, size=(20, 25))
y_test = np.random.randint(0, 1000, size=(20, 25))

# Instantiate and train the model
model = Seq2Seq(xseq_len=25, yseq_len=25, xvocab_size=1000, yvocab_size=1000, emb_dim=128, num_layers=2)
model.train(X_train, y_train, batch_size=16, epochs=400)

# Evaluate the model
model.evaluate(X_test, y_test)
