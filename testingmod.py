import tensorflow as tf
from transformers import AutoTokenizer, TFAutoModelForSequenceClassification
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import csv

# Load the tokenizer and model
tokenizer = AutoTokenizer.from_pretrained('indolem/indobert-base-uncased')
model = TFAutoModelForSequenceClassification.from_pretrained('indolem/indobert-base-uncased', from_pt=True)

# Load the label encoder
with open('./dataset/dataset_sample.csv', 'r', encoding='utf-8') as file:
    reader = csv.reader(file, delimiter='|')
    filtered_rows = [row for row in reader if len(row) == 2 and row[0].strip() != "" and row[1].strip() != ""]

df = pd.DataFrame(filtered_rows, columns=['question', 'answer'])
label_encoder = LabelEncoder()
label_encoder.fit(df['answer'])


input_ids_test = []
attention_masks_test = []
for question in df_test['question']:
    encoded = tokenizer.encode_plus(
        question,
        add_special_tokens=True,
        max_length=128,
        padding='max_length',
        truncation=True,
        return_attention_mask=True,
        return_tensors='tf'
    )
    input_ids_test.append(encoded['input_ids'])
    attention_masks_test.append(encoded['attention_mask'])

input_ids_test = tf.concat(input_ids_test, axis=0)
attention_masks_test = tf.concat(attention_masks_test, axis=0)
labels_test = tf.constant(df_test['encoded_answer'].values)

logits_test = model(input_ids_test, attention_mask=attention_masks_test).logits
preds_test = tf.argmax(logits_test, axis=1, output_type=tf.int32)
test_accuracy_metric.update_state(labels_test, preds_test)

test_accuracy = test_accuracy_metric.result()
print(f"Test accuracy: {test_accuracy:.4f}")

# Interactive loop to get user input and predict
results = []
while True:
    question = input("Enter a question (or 'exit' to quit): ")
    if question.lower() == 'exit':
        break
    answer = predict(question)
    results.append({'Question': question, 'Predicted Answer': answer})
    print(f"Predicted Answer: {answer}")

# Save results to Excel
results_df = pd.DataFrame(results)
results_df.to_excel('./predicted_answers.xlsx', index=False)

print(".xlsx")
