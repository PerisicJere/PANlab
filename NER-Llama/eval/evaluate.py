import json, sys


if len(sys.argv) != 2:
    print("Usage: python script.py language")
    sys.exit(1)

language = sys.argv[1]
if language not in ('en', 'es'):
    print("Language must be 'en' or 'es'")
    sys.exit(1)

input_file = f"NER-Llama/Results/similarities_{language}_results.json"

with open(input_file, "r") as file:
    data = json.load(file)

attribute_counts = {}

for entry in data:
    for attribute in entry['attributes']:
        attr_name = attribute['attribute']
        if attr_name not in attribute_counts:
            attribute_counts[attr_name] = {'true_positives': 0, 'false_positives': 0, 'false_negatives': 0}
        if attribute['cosine_similarity'] == 1.0 and attribute['dataset_value'] == attribute['prediction_value']:
            attribute_counts[attr_name]['true_positives'] += 1
        elif attribute['cosine_similarity'] == 0.0 and attribute['dataset_value'] != attribute['prediction_value']:
            attribute_counts[attr_name]['false_positives'] += 1
        elif attribute['cosine_similarity'] == 0.0 and attribute['dataset_value'] == attribute['prediction_value']:
            attribute_counts[attr_name]['false_negatives'] += 1

attribute_f1_scores = {}
for attr_name, counts in attribute_counts.items():
    true_positives = counts['true_positives']
    false_positives = counts['false_positives']
    false_negatives = counts['false_negatives']
    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) != 0 else 0
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) != 0 else 0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) != 0 else 0
    attribute_f1_scores[attr_name] = f1_score

for attr_name, f1_score in attribute_f1_scores.items():
    print(f"F1 Score for '{attr_name}': {f1_score}")
