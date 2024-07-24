# import library yang di butuhkan
import tensorflow as tf
from transformers import TFT5ForConditionalGeneration, T5Tokenizer
import pandas as pd

# Inisialisasi daftar kosong untuk menyimpan baris yang telah dibersihkan
rows = []