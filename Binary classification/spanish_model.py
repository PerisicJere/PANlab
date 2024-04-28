from transformers import RobertaForSequenceClassification, AutoTokenizer, AdamW
import torch
import pandas as pd
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.metrics import matthews_corrcoef

device = torch.cuda.is_available()
if device is True:
    curr_dev = torch.cuda.current_device()	
    print("Using device:", torch.cuda.get_device_name(curr_dev))
    device = "cuda"
else:
    device = "cpu"
    print("Using device:", device)

model_name = 'cardiffnlp/twitter-xlm-roberta-base-sentiment'
model = RobertaForSequenceClassification.from_pretrained(model_name, num_labels=2,ignore_mismatched_sizes=True)
tokenizer = AutoTokenizer.from_pretrained(model_name)

model.to(device)

train_df = pd.read_csv('PANlab/Binary classification/trainEs.csv')
label_map = {"CONSPIRACY": 0, "CRITICAL": 1}

optimizer = AdamW(model.parameters(), lr=1e-5)
epoch = 0
best_mcc = -1  
patience = 3   
no_improvement = 0 
max_seq_length = 128

train_df, test_df = train_test_split(train_df, test_size=0.2, random_state=42)

test_df, val_df = train_test_split(test_df, test_size=0.5, random_state=42)

while no_improvement < patience:
    epoch += 1
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
    true_labels = []
    predicted_labels = []

    with torch.no_grad():
        for index, row in tqdm(val_df.iterrows(), total=len(val_df), desc=f"Epoch {epoch + 1}/{epochs} - Validation"):
            text = row['Text'].strip()
            label = row['Label']
            label_id = label_map[label]
            inputs = tokenizer(text, return_tensors='pt', max_length=max_seq_length, truncation=True, padding=True)
            inputs = {key: val.to(device) for key, val in inputs.items()}
            outputs = model(**inputs)
            logits = outputs.logits
            predicted = torch.softmax(logits, 1)
            predicted_label_id = torch.argmax(predicted, axis=1).item()
            true_labels.append(label_id)
            predicted_labels.append(predicted_label_id)

    mcc = matthews_corrcoef(true_labels, predicted_labels)
    print(f"Epoch {epoch + 1} - Validation Accuracy: {mcc}")
    if mcc > best_mcc:
        best_mcc = mcc
        torch.save(model.state_dict(), "best_es_model.pt")  
        no_improvement = 0
    else:
        no_improvement += 1

test_texts = test_df['Text'].tolist()
test_labels = test_df['Label'].tolist()

model.load_state_dict(torch.load("best_es_model.pt"))
model.eval()

true_labels = []
predicted_labels = []

with torch.no_grad():
    for text, label in tqdm(zip(test_texts, test_labels), total=len(test_texts), desc="Testing"):
        label_id = label_map[label]
        inputs = tokenizer(text, return_tensors='pt', max_length=max_seq_length, truncation=True, padding=True)
        inputs = {key: val.to(device) for key, val in inputs.items()}
        outputs = model(**inputs)
        logits = outputs.logits
        predicted = torch.softmax(logits, 1)
        predicted_label_id = torch.argmax(predicted, axis=1).item()
        true_labels.append(label_id)
        predicted_labels.append(predicted_label_id)


mcc = matthews_corrcoef(true_labels, predicted_labels)
print("Test Accuracy:", mcc)