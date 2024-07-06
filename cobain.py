import tensorflow as tf
from transformers import BertTokenizer, TFBertForSequenceClassification
import pandas as pd

try:
    # Load the dataset
    df = pd.read_csv('dataset-a.csv', sep='|', on_bad_lines='skip')

    # Print column names to verify
    print(df.columns)  # Ensure 'question|answer' exists and note its exact name if different

    # Split the combined column 'question|answer' into separate 'question' and 'answer' columns
    df['question|answer'] = df['question|answer'].apply(lambda x: x.split('|')[0] if isinstance(x, str) else '')
    df['questionanswer'] = df['question|answer'].apply(lambda x: x.split('|')[1] if isinstance(x, str) and '|' in x else '')

    # Initialize tokenizer
    tokenizer = BertTokenizer.from_pretrained('indobenchmark/indobert-base-p2')

    # Tokenize the input texts
    input_ids = []
    attention_masks = []

    for question, answer in zip(df['question|answer']):
        input_text = f"{question} {answer}"
        encoded = tokenizer.encode_plus(input_text, add_special_tokens=True, max_length=128, pad_to_max_length=True, return_attention_mask=True, truncation=True)
        input_ids.append(encoded['input_ids'])
        attention_masks.append(encoded['attention_mask'])

    input_ids = tf.constant(input_ids)
    attention_masks = tf.constant(attention_masks)
    labels = tf.constant(df['label'].values)  # Assuming 'label' is the target label for classification

    # Load model
    model = TFBertForSequenceClassification.from_pretrained('indobenchmark/indobert-base-p2')

    # Compile and train the model
    model.compile(optimizer=tf.keras.optimizers.Adam(), loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), metrics=['accuracy'])
    model.fit([input_ids, attention_masks], labels, epochs=3, batch_size=32)

    # Save the model
    model.save('indobert_sequence_classification_model')

except pd.errors.ParserError as e:
    print(f"ParserError: {e}")
