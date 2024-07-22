# import yg di butuhkan
import tensorflow as tf
from transformers import TFGPT2LMHeadModel, GPT2Tokenizer
import pandas as pd

# load model dan tokenizer yang telah dilatih
model_path = 'gpt2_model'
model = TFGPT2LMHeadModel.from_pretrained(model_path)
tokenizer = GPT2Tokenizer.from_pretrained(model_path)

# Inisialisasi daftar untuk menyimpan pertanyaan, jawaban, dan akurasi
data = []

