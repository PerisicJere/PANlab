# PANlab

## Models for English dataset
### Model for binary classification
- covid-twitter-bert-v2
#### Idea, and challenges
- Idea was to fine tune the model with the split data
- 80% train, 10% validation, and 10% test data
- Early stoppings is implemented into the model, and it has patience level of 3, the best model is saved
#### Usage
```bash
python3  Binary-classification/english_model.py
```
#### Compariosn of baseline and fine-tuned
| Model | MCC |
--------|------
| Baseline | 0.0226 |
| Fine-Tuned | 0.7554 |
#### Chart
![CT-BERT bar chart](images/MCC_covid-twitter-bert-v2.png)
### Model for named entity recognition (NER)
- Meta-Llama-3-8B-Instruct
#### Idea, and challenges
- Trying to get the best results with prompt engineering
- I think the F1 score is not fair representation of the models accuracy
- Current prompt implemented is the one that performed the best
#### Usage
```bash 
python3 NER-Llama/models/Lama3_ner_en.py
```
- After the model is finished run this command
```bash
python3 NER-Llama/eval/cosines.py
```
- To evaluate your model run
```bash
python3 NER-Llama/eval/evaluate.py
```
#### Results of NER
| Categories | AGENT | FACILITATOR | VICTIM | CAMPAIGNER | OBJECTIVE | NEGATIVE_EFFECT|
|------------|-------|-------------|--------|------------|-----------|----------------|
|F1 scores | 0.2177 | 0.4101 | 0.3462 | 0.3237 | 0.3076 | 0.1633 | 
#### Chart
![Llama3 bar chart](images/llama3-en-ner.png)
## Models for Spanish dataset
### Model for binary classification
- twitter-xlm-roberta-base-sentiment
#### Idea, and challenges
- This model was the best one considering the options
- It is pretrained on twitter data, and it's a cross lingual model
- It performes just a little bit better then random guessing
#### Usage 
```bash
python3 Binary-classification/spanish_model.py
```
#### Compariosn of baseline and fine-tuned
| Model | MCC |
--------|------
| Baseline | 0.0321 |
| Fine-Tuned | 0.5766 |
#### Chart
![RoBERTa](images/Robrta.png)
### Model for named entity reecognition (NER)
- Meta-Llama-3-8B-Instruct
#### Idea, and challenges
- This was probably the hardest subtask
- It underperformed because of the serveral factors
- Next idea is to translate the data to english and the evaluate it
#### Usage
```bash 
python3 NER-Llama/models/Lama3_ner_es.py
```
- After the model is finished run this command
```bash
python3 NER-Llama/eval/cosines.py
```
- To evaluate your model run
```bash
python3 NER-Llama/eval/evaluate.py
```
#### Results of NER
| Categories | AGENT | FACILITATOR | VICTIM | CAMPAIGNER | OBJECTIVE | NEGATIVE_EFFECT|
|------------|-------|-------------|--------|------------|-----------|----------------|
|F1 scores | 0.0314 | 0.0161 | 0.0771 | 0.0393 | 0.0089 | 0.0112 | 
#### Chart
![Llama3-es](images/llama3-es-ner.png)
