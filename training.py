# import torch
# import pandas as pd
# from transformers import BertTokenizer, BertForSequenceClassification, AdamW
# from torch.utils.data import DataLoader, TensorDataset
# from sklearn.model_selection import train_test_split

# # Fungsi untuk membaca dataset
# def read_processed_dataset(file_path):
#     df = pd.read_csv(file_path, sep='|')
#     return df

# # Fungsi untuk mendapatkan DataLoader dari dataset
# def get_dataloader(data, tokenizer, max_length=128, batch_size=32):
#     encoded_data = tokenizer.batch_encode_plus(
#         data['question_answer'].values.tolist(),
#         add_special_tokens=True,
#         max_length=max_length,
#         return_attention_mask=True,
#         pad_to_max_length=True,
#         return_tensors='pt'
#     )
#     dataset = TensorDataset(
#         encoded_data['input_ids'],
#         encoded_data['attention_mask'],
#         torch.tensor(data['label'].values)  # Ganti 'label' dengan kolom target yang sesuai
#     )
#     dataloader = DataLoader(dataset, batch_size=batch_size)
#     return dataloader

# # Fungsi untuk training model
# def train_model(train_dataloader, model, optimizer, num_epochs=3):
#     device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#     model.to(device)
#     model.train()
    
#     for epoch in range(num_epochs):
#         total_loss = 0
#         for batch in train_dataloader:
#             input_ids, attention_mask, labels = batch
#             input_ids, attention_mask, labels = input_ids.to(device), attention_mask.to(device), labels.to(device)
            
#             optimizer.zero_grad()
            
#             outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
#             loss = outputs.loss
#             total_loss += loss.item()
            
#             loss.backward()
#             optimizer.step()
        
#         avg_train_loss = total_loss / len(train_dataloader)
#         print(f'Epoch {epoch + 1}/{num_epochs}, Average Training Loss: {avg_train_loss}')

# # Contoh penggunaan
# def main():
#     try:
#         # Path ke file dataset yang sudah dipreprocessing
#         processed_file = 'dataset_clean.csv'
#         df = read_processed_dataset(processed_file)
        
#         # Inisialisasi tokenizer dan model BERT
#         tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
#         model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2)  # Sesuaikan dengan jumlah label/target Anda
        
#         # Bagi dataset menjadi data latih dan data validasi
#         train_df, val_df = train_test_split(df, test_size=0.2, random_state=42)
        
#         # Dapatkan DataLoader untuk data latih
#         train_dataloader = get_dataloader(train_df, tokenizer)
        
#         # Inisialisasi optimizer
#         optimizer = AdamW(model.parameters(), lr=2e-5, eps=1e-8)
        
#         # Training model
#         train_model(train_dataloader, model, optimizer)
    
#     except Exception as e:
#         print(f"Terjadi kesalahan: {e}")

# if __name__ == "__main__":
#     main()





# tester kedua
import pandas as pd
from transformers import BertTokenizer, BertForSequenceClassification, AdamW
import torch
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split

# Baca dataset
df = pd.read_csv('dataset_clean.csv')

# Tambahkan kolom label sesuai dengan kebutuhan
df['label'] = [0] * len(df)  # Ganti dengan label yang sesuai

# Split dataset menjadi data latih dan data uji
train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)

# Inisialisasi tokenizer BERT
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# Tokenisasi teks menggunakan tokenizer BERT
train_encodings = tokenizer(train_df['question|answer'].tolist(), truncation=True, padding=True)
test_encodings = tokenizer(test_df['question|answer'].tolist(), truncation=True, padding=True)

# Buat dataset Tensor untuk PyTorch
train_dataset = TensorDataset(
    torch.tensor(train_encodings['input_ids']),
    torch.tensor(train_encodings['attention_mask']),
    torch.tensor(train_df['label'].tolist())
)

test_dataset = TensorDataset(
    torch.tensor(test_encodings['input_ids']),
    torch.tensor(test_encodings['attention_mask']),
    torch.tensor(test_df['label'].tolist())
)

# Tentukan batch size
batch_size = 16

# Buat DataLoader untuk data latih dan data uji
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# Inisialisasi model BERT untuk fine-tuning
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2)  # Sesuaikan dengan jumlah label/target Anda

# Pilih optimizer
optimizer = AdamW(model.parameters(), lr=2e-5)

# Fungsi untuk evaluasi
def evaluate(model, dataloader):
    model.eval()
    total_eval_accuracy = 0
    total_eval_loss = 0
    nb_eval_steps = 0
    
    for batch in dataloader:
        batch = tuple(t.to(device) for t in batch)
        inputs = {'input_ids': batch[0], 'attention_mask': batch[1], 'labels': batch[2]}
        with torch.no_grad():        
            outputs = model(**inputs)
        loss, logits = outputs.loss, outputs.logits
        total_eval_loss += loss.item()
        logits = logits.detach().cpu().numpy()
        label_ids = inputs['labels'].to('cpu').numpy()
        total_eval_accuracy += (logits.argmax(axis=-1) == label_ids).mean()
    
    avg_val_accuracy = total_eval_accuracy / len(dataloader)
    avg_val_loss = total_eval_loss / len(dataloader)
    return avg_val_accuracy, avg_val_loss

# Tentukan device (GPU jika tersedia)
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
model.to(device)

# Training loop
epochs = 3
for epoch in range(epochs):
    model.train()
    total_train_loss = 0
    for batch in train_loader:
        batch = tuple(t.to(device) for t in batch)
        inputs = {'input_ids': batch[0], 'attention_mask': batch[1], 'labels': batch[2]}
        optimizer.zero_grad()
        outputs = model(**inputs)
        loss = outputs.loss
        total_train_loss += loss.item()
        loss.backward()
        optimizer.step()
    
    avg_train_loss = total_train_loss / len(train_loader)
    print(f'Epoch {epoch + 1}/{epochs}, Average Training Loss: {avg_train_loss}')
    
    # Evaluasi setiap epoch
    val_accuracy, val_loss = evaluate(model, test_loader)
    print(f'Epoch {epoch + 1}/{epochs}, Validation Accuracy: {val_accuracy}, Validation Loss: {val_loss}')

# Simpan model setelah pelatihan
model.save_pretrained('output_dir/model_bert')
