import pandas as pd
from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments

# Load pretrained DialoGPT model and tokenizer
model_name = "microsoft/DialoGPT-medium"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

# Load CSV dataset
dataset_path = "dataset.csv"
df = pd.read_csv(dataset_path)

# Preprocess dataset
conversations = [{"speaker": row['speaker'], "text": row['text']} for _, row in df.iterrows()]

# Prepare input for training
inputs = tokenizer([c['text'] for c in conversations], padding=True, return_tensors="pt")

# Define training arguments
training_args = TrainingArguments(
    per_device_train_batch_size=4,
    num_train_epochs=3,
    logging_dir='./logs',
    logging_steps=100,
    save_steps=1000,
    evaluation_strategy="epoch",
    overwrite_output_dir=True,
    learning_rate=5e-5,
)

# Create Trainer instance
trainer = Trainer(
    model=model,
    args=training_args,
)

# Define function to format datasets
def format_dataset(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True)

# Train the model
trainer.train(
    train_dataset=format_dataset(inputs),
)

# Save the model
model.save_pretrained("./dialogpt-trained")
