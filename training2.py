import pandas as pd
from transformers import T5Tokenizer, T5ForConditionalGeneration, AdamW
import torch
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split

# Baca dataset
df = pd.read_csv('dataset_clean.csv')

# Split dataset menjadi data latih dan data uji
train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)

# Inisialisasi tokenizer T5
tokenizer = T5Tokenizer.from_pretrained('t5-small')

# Buat dataset untuk teks-to-teks
class QADataset(Dataset):
    def __init__(self, data, tokenizer, max_source_length=512, max_target_length=32):
        self.data = data
        self.tokenizer = tokenizer
        self.max_source_length = max_source_length
        self.max_target_length = max_target_length
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        source_text = self.data.iloc[idx]['question|answer']
        target_text = self.data.iloc[idx]['target_text']  # Ganti dengan kolom yang sesuai
        source = self.tokenizer.encode_plus(source_text, max_length=self.max_source_length, padding='max_length', truncation=True, return_tensors='pt')
        target = self.tokenizer.encode_plus(target_text, max_length=self.max_target_length, padding='max_length', truncation=True, return_tensors='pt')
        return {
            'input_ids': source['input_ids'].flatten(),
            'attention_mask': source['attention_mask'].flatten(),
            'labels': target['input_ids'].flatten(),
            'decoder_attention_mask': target['attention_mask'].flatten()
        }

# Buat dataset untuk training dan testing
train_dataset = QADataset(train_df, tokenizer)
test_dataset = QADataset(test_df, tokenizer)

# Tentukan batch size
batch_size = 16

# Buat DataLoader untuk data latih dan data uji
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# Inisialisasi model T5 untuk fine-tuning
model = T5ForConditionalGeneration.from_pretrained('t5-small')

# Pilih optimizer
optimizer = AdamW(model.parameters(), lr=2e-5)

# Training loop
epochs = 3
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
model.to(device)

for epoch in range(epochs):
    model.train()
    total_train_loss = 0
    for batch in train_loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        optimizer.zero_grad()
        outputs = model(**batch, labels=batch['labels'])
        loss = outputs.loss
        total_train_loss += loss.item()
        loss.backward()
        optimizer.step()
    
    avg_train_loss = total_train_loss / len(train_loader)
    print(f'Epoch {epoch + 1}/{epochs}, Average Training Loss: {avg_train_loss}')
    
    # Evaluasi setiap epoch
    model.eval()
    total_eval_loss = 0
    for batch in test_loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        with torch.no_grad():
            outputs = model(**batch, labels=batch['labels'])
        loss = outputs.loss
        total_eval_loss += loss.item()
    
    avg_val_loss = total_eval_loss / len(test_loader)
    print(f'Epoch {epoch + 1}/{epochs}, Validation Loss: {avg_val_loss}')

# Simpan model setelah pelatihan
model.save_pretrained('output_dir/model_t5')
