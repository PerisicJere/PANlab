import pandas as pd
from transformers import BertTokenizer, BertForTokenClassification
from transformers import pipeline
from sklearn.metrics import f1_score, classification_report
from tqdm import tqdm
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model_name = "bert-base-multilingual-uncased"
tokenizer = BertTokenizer.from_pretrained(model_name)
model = BertForTokenClassification.from_pretrained(model_name).to(device)
ner_pipeline = pipeline("ner", model=model, tokenizer=tokenizer)
data = pd.read_csv('annotationsEn.csv')

tokenized_data = []

unique_ids = set(data['id'])
for unique_id in tqdm(unique_ids, desc="Tokenizing data"):
    subset_data = data[data['id'] == unique_id]
    text = subset_data['text'].iloc[0]
    
    tokens = tokenizer.tokenize(text)
    
    for _, row in subset_data.iterrows():
        tokenized_data.append({'tokens': tokens, 'category': row['category']})

true_labels = []
predicted_labels = []

for item in tqdm(tokenized_data, desc="Predicting labels"):
    tokens = item['tokens']
    category = item['category']
    
    predicted_tags = ner_pipeline(" ".join(tokens))
    predicted_labels.extend([entity['entity'] for entity in predicted_tags])
    
    true_labels.extend([category] * len(predicted_tags))

f1_micro = f1_score(true_labels, predicted_labels, average=None, zero_division="warn")
class_report = classification_report(true_labels, predicted_labels, zero_division="warn")

print("Micro F1 score:", f1_micro)
print("Classification Report:")
print(class_report)
