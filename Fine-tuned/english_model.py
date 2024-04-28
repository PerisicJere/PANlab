from transformers import BertForSequenceClassification, BertTokenizer, AdamW
import torch
import pandas as pd
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

device = torch.cuda.is_available()
if device is True:
    curr_dev = torch.cuda.current_device()	
    print("Using device:", torch.cuda.get_device_name(curr_dev))
    device = "cuda"
else:
    device = "cpu"
    print("Using device:", device)

model_name = 'digitalepidemiologylab/covid-twitter-bert-v2'
model = BertForSequenceClassification.from_pretrained(model_name, num_labels=2)
tokenizer = BertTokenizer.from_pretrained(model_name)

model.to(device)

train_df = pd.read_csv('trainEn.csv')
label_map = {"CONSPIRACY": 0, "CRITICAL": 1}

optimizer = AdamW(model.parameters(), lr=1e-5)
epochs = 3
max_seq_length = 128

train_df, test_df = train_test_split(train_df, test_size=0.1, random_state=42)

train_df, val_df = train_test_split(train_df, test_size=0.11, random_state=42)

for epoch in range(epochs):
    model.train()
    for index, row in tqdm(train_df.iterrows(), total=len(train_df), desc=f"Epoch {epoch + 1}/{epochs} - Training"):
        text = row['Text'].strip()
        label = row['Label']
        label_id = label_map[label]
        inputs = tokenizer(text, return_tensors='pt', max_length=max_seq_length, truncation=True, padding=True)
        inputs = {key: val.to(device) for key, val in inputs.items()}
        outputs = model(**inputs, labels=torch.tensor([label_id]).to(device))
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

    model.eval()
    correct_predictions = 0
    total_predictions = 0

    with torch.no_grad():
        for index, row in tqdm(val_df.iterrows(), total=len(val_df), desc=f"Epoch {epoch + 1}/{epochs} - Validation"):
            text = row['Text'].strip()
            label = row['Label']
            label_id = label_map[label]
            inputs = tokenizer(text, return_tensors='pt', max_length=max_seq_length, truncation=True, padding=True)
            inputs = {key: val.to(device) for key, val in inputs.items()}
            outputs = model(**inputs)
            logits = outputs.logits
            _, predicted = torch.max(logits, 1)
            total_predictions += 1
            if predicted == label_id:
                correct_predictions += 1

    val_accuracy = correct_predictions / total_predictions
    print(f"Epoch {epoch + 1}/{epochs} - Validation Accuracy: {val_accuracy}")


test_texts = test_df['Text'].tolist()
test_labels = test_df['Label'].tolist()


model.eval()
correct_predictions = 0
total_predictions = 0

with torch.no_grad():
    for text, label in tqdm(zip(test_texts, test_labels), total=len(test_texts), desc="Testing"):
        label_id = label_map[label]
        inputs = tokenizer(text, return_tensors='pt', max_length=max_seq_length, truncation=True, padding=True)
        inputs = {key: val.to(device) for key, val in inputs.items()}
        outputs = model(**inputs)
        logits = outputs.logits
        _, predicted = torch.max(logits, 1)
        total_predictions += 1
        if predicted == label_id:
            correct_predictions += 1

test_accuracy = correct_predictions / total_predictions
print("Test Accuracy:", test_accuracy)
