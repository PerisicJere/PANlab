from transformers import BertForSequenceClassification, BertTokenizer
import torch, tqdm
import pandas as pd


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
label_map = {"CONSPIRACY": 0, "CRITICAL": 1}

model.eval()
correct_predictions = 0
total_predictions = 0

with torch.no_grad():
    for index,row in base_df.iterrows():
        text = row['Text']
        label = row['Label']
        label_id = label_map[label]
        inputs = tokenizer(text, return_tensors='pt', max_length=128, truncation=True, padding=True)
        inputs = {key: val.to(device) for key, val in inputs.items()}
        outputs = model(**inputs)
        logits = outputs.logits
        _, predicted = torch.max(logits, 1)
        total_predictions += 1
        if predicted == label_id:
            correct_predictions += 1

base_accuracy = correct_predictions / total_predictions
print(f"Baseline Accuracy: {base_accuracy}")
