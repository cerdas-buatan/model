import torch
from torch import nn, optim
from torch.utils.data import DataLoader, Dataset
from transformers import GPT2Tokenizer, GPT2LMHeadModel

class TextDataset(Dataset):
    def __init__(self, texts, tokenizer, max_length):
        self.texts = texts
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        encodings = self.tokenizer(self.texts[idx], truncation=True, padding='max_length', max_length=self.max_length, return_tensors='pt')
        return encodings.input_ids[0], encodings.attention_mask[0]

def train(model, dataloader, optimizer, scheduler, device):
    model.train()
    total_loss = 0
    for batch in dataloader:
        inputs, masks = batch
        inputs = inputs.to(device)
        masks = masks.to(device)

        outputs = model(input_ids=inputs, attention_mask=masks, labels=inputs)
        loss = outputs.loss
        loss.backward()
        
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()
        
        total_loss += loss.item()
    return total_loss / len(dataloader)

def main():
    # Load dataset
    import pandas as pd
    df = pd.read_csv('dataset/a-backup.csv')
    texts = df['text_column'].tolist()

    # Initialize tokenizer and model
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    model = GPT2LMHeadModel.from_pretrained('gpt2')
    model = model.to(device)

    # Create dataset and dataloader
    dataset = TextDataset(texts, tokenizer, max_length=512)
    dataloader = DataLoader(dataset, batch_size=8, shuffle=True)

    # Training setup
    optimizer = optim.AdamW(model.parameters(), lr=5e-5)
    scheduler = optim.lr_scheduler.OneCycleLR(optimizer, max_lr=5e-5, steps_per_epoch=len(dataloader), epochs=3)

    # Training loop
    epochs = 3
    for epoch in range(epochs):
        loss = train(model, dataloader, optimizer, scheduler, device)
        print(f'Epoch {epoch+1}, Loss: {loss}')

    # Save the model
    model.save_pretrained('trained_model')

if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    main()
