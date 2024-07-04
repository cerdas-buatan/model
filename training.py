import torch
import pandas as pd
from transformers import BertTokenizer, BertForSequenceClassification, AdamW
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split

# Fungsi untuk membaca dataset
def read_processed_dataset(file_path):
    df = pd.read_csv(file_path, sep='|')
    return df

# Fungsi untuk mendapatkan DataLoader dari dataset
def get_dataloader(data, tokenizer, max_length=128, batch_size=32):
    encoded_data = tokenizer.batch_encode_plus(
        data['question_answer'].values.tolist(),
        add_special_tokens=True,
        max_length=max_length,
        return_attention_mask=True,
        pad_to_max_length=True,
        return_tensors='pt'
    )
    dataset = TensorDataset(
        encoded_data['input_ids'],
        encoded_data['attention_mask'],
        torch.tensor(data['label'].values)  # Ganti 'label' dengan kolom target yang sesuai
    )
    dataloader = DataLoader(dataset, batch_size=batch_size)
    return dataloader

# Fungsi untuk training model
def train_model(train_dataloader, model, optimizer, num_epochs=3):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    model.train()
    
    for epoch in range(num_epochs):
        total_loss = 0
        for batch in train_dataloader:
            input_ids, attention_mask, labels = batch
            input_ids, attention_mask, labels = input_ids.to(device), attention_mask.to(device), labels.to(device)
            
            optimizer.zero_grad()
            
            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            total_loss += loss.item()
            
            loss.backward()
            optimizer.step()
        
        avg_train_loss = total_loss / len(train_dataloader)
        print(f'Epoch {epoch + 1}/{num_epochs}, Average Training Loss: {avg_train_loss}')

# Contoh penggunaan
def main():
    try:
        # Path ke file dataset yang sudah dipreprocessing
        processed_file = 'dataset_clean.csv'
        df = read_processed_dataset(processed_file)
        
        # Inisialisasi tokenizer dan model BERT
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2)  # Sesuaikan dengan jumlah label/target Anda
        
        # Bagi dataset menjadi data latih dan data validasi
        train_df, val_df = train_test_split(df, test_size=0.2, random_state=42)
        
        # Dapatkan DataLoader untuk data latih
        train_dataloader = get_dataloader(train_df, tokenizer)
        
        # Inisialisasi optimizer
        optimizer = AdamW(model.parameters(), lr=2e-5, eps=1e-8)
        
        # Training model
        train_model(train_dataloader, model, optimizer)
    
    except Exception as e:
        print(f"Terjadi kesalahan: {e}")

if __name__ == "__main__":
    main()
