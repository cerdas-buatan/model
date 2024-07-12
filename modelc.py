import pandas as pd
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments

# Load your clean.csv file
df = pd.read_csv('clean.csv')

# Check the first few rows of the dataframe
df.head()

# Load the tokenizer and model
tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-medium")
model = AutoModelForCausalLM.from_pretrained("microsoft/DialoGPT-medium")


#def tokenize_function(examples):
    #return tokenizer(examples["input"], padding="max_length", truncation=True, max_length=128)
def tokenize_function(examples):
    return tokenizer(examples["question"], padding="max_length", truncation=True, max_length=128)

# Tokenize the input and response columns
train_data = df[['question', 'answer']].apply(lambda row: {'question': row['question'], 'answer': row['answer']}, axis=1)
train_data = train_data.apply(tokenize_function)

training_args = TrainingArguments(
    output_dir="./results",
    num_train_epochs=10,
    #per_device_train_batch_size=10,
    #per_device_eval_batch_size=10,
    per_device_train_batch_size=20,
    per_device_eval_batch_size=20,
    warmup_steps=500,
    weight_decay=0.01,
    logging_dir="./logs",
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_data,
)

trainer.train()

# Generate a response for a new input
new_user_input = "How are you?"
input_ids = tokenizer.encode(new_user_input + tokenizer.eos_token, return_tensors="pt")
chat_history_ids = model.generate(input_ids, max_length=1500, pad_token_id=tokenizer.eos_token_id)

# Decode the response
response = tokenizer.decode(chat_history_ids[:, input_ids.shape[-1]:][0], skip_special_tokens=True)
print(response)
