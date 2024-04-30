import json
import sys
import numpy as np

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
confusion_matrix = np.zeros((2, 2), dtype=int)  

# process similarity results
for entry in data:
    for attribute in entry['attributes']:
        attr_name = attribute['attribute']
        actual = attribute['dataset_value']
        predicted = attribute['prediction_value']
        similarity = attribute['cosine_similarity']

        if actual == predicted:
            actual_class = 1
        else:
            actual_class = 0

        if similarity == 1.0:
            predicted_class = 1
        else:
            predicted_class = 0

        confusion_matrix[actual_class][predicted_class] += 1

        if attr_name not in attribute_counts:
            attribute_counts[attr_name] = {'true_positives': 0, 'false_positives': 0, 'false_negatives': 0}
        
        if similarity == 1.0 and actual == predicted:
            attribute_counts[attr_name]['true_positives'] += 1
        elif similarity == 0.0 and actual != predicted:
            attribute_counts[attr_name]['false_positives'] += 1
        elif similarity == 0.0 and actual == predicted:
            attribute_counts[attr_name]['false_negatives'] += 1

# calculate F1 scores for attributes
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
    print(f"F1 Score for '{attr_name}': {f1_score:.4f}")

print("Confusion Matrix:")
print(confusion_matrix)
