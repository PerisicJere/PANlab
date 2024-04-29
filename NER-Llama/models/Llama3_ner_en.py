from transformers import pipeline
import transformers
import torch
import json, re


model_id = "meta-llama/Meta-Llama-3-8B-Instruct"

pipeline = transformers.pipeline(
    "text-generation",
    model=model_id,
    model_kwargs={"torch_dtype": torch.bfloat16},
    device_map="auto"
)
def get_ner(pipeline):
    
    with open('NER-Llama/datasets/dataset_en_train.json', 'r') as json_file:
        data = json.load(json_file)
    for item in data:
        ids = item['id']
        text = item['text']
        messages = [
            {"role": "system", "content": "Task: Extract specific elements—AGENT, FACILITATOR, VICTIM, CAMPAIGNER, OBJECTIVE, and NEGATIVE_EFFECT—from a given text, omitting any elements that are not explicitly mentioned. Instructions: Provide the identified elements directly from the input text without alterations. Only include elements that are mentioned explicitly."},
            {"role": "user", "content": text},
        ]

        prompt = pipeline.tokenizer.apply_chat_template(
                messages, 
                tokenize=False, 
                add_generation_prompt=True
        )

        terminators = [
            pipeline.tokenizer.eos_token_id,
            pipeline.tokenizer.convert_tokens_to_ids("<|eot_id|>")
        ]

        outputs = pipeline(
            prompt,
            max_new_tokens=256,
            eos_token_id=terminators,
            do_sample=True,
            temperature=0.6,
            top_p=0.9,
        )
        generated_text = outputs[0]["generated_text"][len(prompt):]

        extracted_elements = {
            "id": ids,
        }
        split_text = generated_text.split("\n")
        for entry in split_text:
            if ":" in entry:
                parts = entry.split(":", 1)
                if len(parts) == 2:
                    key, value = parts
                    key = re.sub(r'\*', '', key).strip()
                    value = value.strip()
                    extracted_elements[key] = value
                else:
                    print("Warning: Entry '{}' does not contain exactly one colon. Skipping.".format(entry))

        with open("NER-Llama/Results/predictions_en.json", "a") as json_file:
            json.dump(extracted_elements, json_file, indent=4)
            json_file.write(",\n")
        
        
get_ner(pipeline)