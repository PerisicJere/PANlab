from transformers import BertForSequenceClassification, BertTokenizer
from sklearn.metrics import accuracy_score,f1_score
import torch, tqdm
import pandas as pd
from prettytable import PrettyTable

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if device is True:
    curr_dev = torch.cuda.current_device()
    print("Using device:", torch.cuda.get_device_name(curr_dev))
else:
    print("Using device:", device)

model_name = 'digitalepidemiologylab/covid-twitter-bert-v2'
model = BertForSequenceClassification.from_pretrained(model_name, num_labels=2)
tokenizer = BertTokenizer.from_pretrained(model_name)

model.to(device)

base_df = pd.read_csv('trainEn.csv')
label2id = {"CONSPIRACY": 0, "CRITICAL": 1}

myTable = PrettyTable(["Conspiracy", "Critical"])

model.eval()
true_labels = []
predicted_labels = []

with torch.no_grad():
    for index,row in base_df.iterrows():
        text = row['Text']
        label = row['Label']
        label_id = label2id.get(label)
        inputs = tokenizer(text, return_tensors='pt', max_length=128, truncation=True, padding=True)
        inputs = {key: val.to(device) for key, val in inputs.items()}
        outputs = model(**inputs)
        logits = outputs.logits
        predicted = torch.softmax(logits, 1)
        predicted_label_id = torch.argmax(predicted, axis=1).item()
        true_labels.append(label_id)
        predicted_labels.append(predicted_label_id)

f1_scores = f1_score(true_labels, predicted_labels, average=None)
myTable.add_row(f1_scores)
print(myTable)
base_accuracy = accuracy_score(true_labels, predicted_labels)
print(f"Baseline Accuracy: {base_accuracy}")
