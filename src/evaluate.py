import pandas as pd
import numpy as np
import torch

from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import Dataset

from dataloader import load_abnormal_descriptions, load_normal_descriptions, match_anomalies_with_descriptions

def generate_predictions_with_prompt(model, tokenizer, dataset):
    predictions = []
    for example in dataset:
        # Construct the prompt including start and end points
        prompt = f"start: {example['start']}, end: {example['end']}, reason:"
        
        # Generate response using the fine-tuned model
        input_ids = tokenizer.encode(prompt, return_tensors="pt")
        output = model.generate(input_ids=input_ids, max_length=100, num_return_sequences=1)
        prediction = tokenizer.decode(output[0], skip_special_tokens=True)
        
        # Append the prediction in the specified format
        predictions.append(prediction)
    
    return predictions

def main(): 
    # Load data
    df = pd.read_csv('')
    abnormal_descriptions = load_abnormal_descriptions('/path/to/abnormal_descriptions.txt')
    normal_description = load_normal_descriptions('/path/to/normal_description.txt')

    # Initialize model and tokenizer
    model = AutoModelForCausalLM.from_pretrained("facebook/opt-350m")
    tokenizer = AutoTokenizer.from_pretrained("facebook/opt-350m")

    # Construct dataset
    instructions = []
    outputs = []
    for desc in abnormal_descriptions:
        instructions.append(f"Given the values provided, please classify the which interval is 'abnormal'. Please provide your reasoning based on the attributes' values and their expected pattern.")
        outputs.append(normal_description)
    dataset = Dataset.from_dict({"instruction": instructions, "output": outputs})

    # Generate predictions
    # predictions = generate_predictions_with_prompt(model, tokenizer, dataset)

    # Print predictions
    for prediction in predictions:
        print(prediction)

if __name__ == "__main__":
    main()
