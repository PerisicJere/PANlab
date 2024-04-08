from transformers import AutoModelForSequenceClassification, AutoTokenizer
from sklearn.metrics import matthews_corrcoef, accuracy_score
import torch, tqdm
import pandas as pd
from prettytable import PrettyTable

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if device is True:
    curr_dev = torch.cuda.current_device()
    print("Using device:", torch.cuda.get_device_name(curr_dev))
else:
    print("Using device:", device)

model_name = 'cardiffnlp/twitter-xlm-roberta-base-sentiment'
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2,ignore_mismatched_sizes=True)
tokenizer = AutoTokenizer.from_pretrained(model_name)

model.to(device)

label2id = {"CONSPIRACY": 0, "CRITICAL": 1}

def preprocess_text(text):
    return tokenizer.encode_plus(text, add_special_tokens=True, padding='max_length', max_length=128, return_tensors='pt', truncation=True)

test_df = pd.read_csv('trainEs.csv')

predicted_labels = []
true_labels = []

for index, row in test_df.iterrows():
    text = row['Text']
    label = row['Label'] 

    inputs = preprocess_text(text)
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)

    logits = outputs.logits
    predicted_label = torch.argmax(logits, dim=1).item()

    predicted_labels.append(predicted_label)
    true_labels.append(label2id[label]) 



mcc = matthews_corrcoef(true_labels, predicted_labels)

myTable = PrettyTable(["Metric", "Score"])
myTable.add_row(["Matthews Correlation Coefficient", mcc])
print(myTable)

