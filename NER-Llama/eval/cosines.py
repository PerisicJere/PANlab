import json, sys
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer


if len(sys.argv) != 2:
    print("Usage: python script.py language")
    sys.exit(1)

language = sys.argv[1]
if language not in ('en', 'es'):
    print("Language must be 'en' or 'es'")
    sys.exit(1)
    
dataset = f'NER-Llama/datasets/dataset_{language}_train.json'
predictions = f'NER-Llama/Results/predictions_{language}.json'
output_file = f'NER-Llama/Results/similarities_{language}_results.json'

with open(dataset, 'r') as dataset_file:
    dataset_data = json.load(dataset_file)

with open(predictions, 'r') as predictions_file:
    predictions_data = json.load(predictions_file)

tfidf_vectorizer = TfidfVectorizer()

def compute_cosine_similarity(string1, string2):
    try:
        tfidf_matrix = tfidf_vectorizer.fit_transform([string1, string2])
        cosine_sim = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])
        return cosine_sim[0][0]
    except ValueError:
        return 0.0

attributes = ["AGENT", "FACILITATOR", "VICTIM", "CAMPAIGNER", "OBJECTIVE", "NEGATIVE_EFFECT"]

results = []

for data_point in dataset_data:
    data_point_id = data_point['id']
    data_point_results = {"id": data_point_id, "attributes": []}
    
    prediction_data_point = next((p for p in predictions_data if p['id'] == data_point_id), None)
    if prediction_data_point is None:
        print(f"No prediction found for ID: {data_point_id}")
        continue
    
    for attribute in attributes:
        dataset_value = next((anno.get("span_text") for anno in data_point.get("annotations", []) if anno.get("category") == attribute), None)
        prediction_value = prediction_data_point.get(attribute, '')
        
        if dataset_value is None and prediction_value == '':
            similarity = 1.0
        elif dataset_value is None:
            similarity = 1.0 if compute_cosine_similarity("None mentioned", prediction_value) > 0.5 else 0.0
        else:
            similarity = 1.0 if compute_cosine_similarity(dataset_value, prediction_value) > 0.4 else 0.0
        
        data_point_results["attributes"].append({
            "attribute": attribute,
            "dataset_value": dataset_value if dataset_value is not None else 'None mentioned',
            "prediction_value": prediction_value,
            "cosine_similarity": similarity
        })

    results.append(data_point_results)


with open(output_file, 'w') as json_output:
    json.dump(results, json_output, indent=4)

print(f"Results saved to: {output_file}")
