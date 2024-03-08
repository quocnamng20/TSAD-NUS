import pandas as pd
import numpy as np
import torch

from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import Dataset

from dataloader import load_abnormal_descriptions, load_normal_descriptions, match_anomalies_with_descriptions

def construct_multi_general_test(df, normal_description):
    instructions = []
    answers = []
    
    # Number of attributes to describe, excluding the last 3 columns
    num_attributes = len(df.columns) - 3

    # Loop through the DataFrame in steps of 50 rows
    for start_index in range(0, len(df), 50):
        # Ensure we don't go out of bounds for the last set
        end_index = min(start_index + 50, len(df))
        
        # Generate attribute descriptions by compiling values from each attribute within the interval
        attributes_description = []
        for k in range(1, num_attributes):
            # Extract values for the attribute in the current interval
            values = df.iloc[start_index:end_index, k].tolist()
            # Construct description for the current attribute
            attribute_description = f"Attribute {k+1} values are {', '.join(map(str, values))}"
            attributes_description.append(attribute_description)
        
        # Join all attribute descriptions
        full_description = '. '.join(attributes_description)
        
        # Construct the instruction text
        instruction = (f"Consider the following attributes from a time series data: {full_description}. "
                       "Given the values provided, please classify which interval is 'normal', which interval is 'abnormal'. Please provide your reasoning based on the attributes' values and their expected pattern.\n")
        
        # Append the constructed texts to their respective lists
        instructions.append(instruction)
        answers.append(f"The normal pattern is {normal_description}")

    return instructions, answers

def construct_uni_general_test(df, normal_description):
    instructions = []
    answers = []
    
    # Number of attributes to describe, excluding the last 3 columns

    # Loop through the DataFrame in steps of 50 rows
    for start_index in range(0, len(df), 50):
        # Ensure we don't go out of bounds for the last set
        end_index = min(start_index + 50, len(df))
        
        # Generate attribute descriptions by compiling values from each attribute within the interval
        attributes_description = []

        values = df.iloc[start_index:end_index]['value'].tolist()
        # Construct description for the current attribute
        attribute_description = f"Values are {', '.join(map(str, values))}"
        attributes_description.append(attribute_description)
        
        # Join all attribute descriptions
        full_description = '. '.join(attributes_description)
        
        # Construct the instruction text
        instruction = (f"Consider the following attributes from a time series data interval: {full_description}. "
                       "Given the values provided, please classify which interval is 'normal', which interval is 'abnormal'. Please provide your reasoning based on the values and their expected pattern.\n")
        
        # Append the constructed texts to their respective lists
        instructions.append(instruction)
        answers.append(f"The normal pattern is {normal_description}")

    return instructions, answers

# Function to generate predictions
def generate_predictions(model, tokenizer, instructions):
    predictions = []
    for instruction in instructions:
        input_ids = tokenizer.encode(instruction, return_tensors="pt")
        output = model.generate(input_ids=input_ids, max_length=100, num_return_sequences=1)
        prediction_text = tokenizer.decode(output[0], skip_special_tokens=True)
        predictions.append(prediction_text)
    return predictions

def generate_testset(mode):
    if mode == 'uni':
        df = pd.read_csv('/home/tri/llms-anomaly-detection/TSAD-NUS/Dataset/Univariate_data/145_UCR_Anomaly_Lab2Cmac011215EPG1_5000_17210_17260/145_UCR_Anomaly_Lab2Cmac011215EPG1_5000_17210_17260.txt', header=None, names=['value'])
        df = df[5000:]
        abnormal_descriptions = load_abnormal_descriptions('/home/tri/llms-anomaly-detection/TSAD-NUS/Dataset/Univariate_data/145_UCR_Anomaly_Lab2Cmac011215EPG1_5000_17210_17260/description_abnormal.txt')
        normal_description = load_normal_descriptions('/home/tri/llms-anomaly-detection/TSAD-NUS/Dataset/Univariate_data/145_UCR_Anomaly_Lab2Cmac011215EPG1_5000_17210_17260/description_normal.txt')

    else:
        df = pd.read_csv('/home/tri/llms-anomaly-detection/TSAD-NUS/Dataset/Multivariate_data/Gen_1/gen_2.csv')
        df = df[150:]
        abnormal_descriptions = load_abnormal_descriptions('/home/tri/llms-anomaly-detection/TSAD-NUS/Dataset/Multivariate_data/Gen_1/description_abnormal.txt')
        normal_description = load_normal_descriptions('/home/tri/llms-anomaly-detection/TSAD-NUS/Dataset/Multivariate_data/Gen_1/description_normal.txt')
    
    df['description'] = normal_description

    for desc in abnormal_descriptions:
        df.loc[df.index.isin(range(desc['start'], desc['end'] + 1)), 'description'] = desc['reason']
    
    if mode == 'uni':
        instruction, outputs = construct_uni_general_test(df, normal_description)
    else:
        instruction, outputs = construct_uni_general_test(df, normal_description)

    return pd.DataFrame({'question': instruction, 'answer': outputs})


def main(): 

    # Initialize model and tokenizer
    model = AutoModelForCausalLM.from_pretrained("facebook/opt-350m")
    tokenizer = AutoTokenizer.from_pretrained("facebook/opt-350m")

    # Construct dataset
    instructions = []
    outputs = []

    dataset = generate_testset('uni')
    # Generate predictions
    predictions = generate_predictions(model, tokenizer, dataset['question'])

    # Print predictions
    # for prediction in predictions:
    #     print(prediction)

if __name__ == "__main__":
    main()
