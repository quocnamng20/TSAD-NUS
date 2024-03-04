import os
import pandas as pd
import numpy as np
import torch

from transformers import GPT2Tokenizer, GPT2LMHeadModel, TrainingArguments, Trainer, GPT2ForSequenceClassification, pipeline, AutoModelForCausalLM, AutoTokenizer, AutoModelForSeq2SeqLM, AdamW, get_linear_schedule_with_warmup
from torch.utils.data import DataLoader
from datasets import load_metric, load_dataset, Dataset
from sklearn.model_selection import train_test_split
from trl import SFTTrainer, DataCollatorForCompletionOnlyLM

def load_abnormal_descriptions(file_path):
    descriptions = []
    structured_description = []
    with open(file_path, 'r') as file:
        for line in file:
            parts = line.strip().split(', ')
            # print(parts)
            # if len(parts) == 3:
            # start, end, reason = parts
            # descriptions.append({'start': int(start), 'end': int(end), 'reason': reason})
            structured_description = {
                'start': int(parts[0].split(': ')[1]),
                'end': int(parts[1].split(': ')[1]),
                'reason': ', '.join(parts[2:]).split(': ', 1)[1]
            }
            descriptions.append(structured_description)
    return descriptions

def load_normal_descriptions(file_path):
    with open(file_path, 'r') as file:
        normal_description = file.read().strip()
    return normal_description

def match_anomalies_with_descriptions(time_series_data, descriptions):
    anomalies_output = []
    for description in descriptions:
        start, end = description['start'], description['end']
        # Check if there is any anomaly within this interval
        if time_series_data[(time_series_data.index >= start) & (time_series_data.index <= end)]['label'].any():
            anomalies_output.append(description)
    return anomalies_output

# Load your data
df = pd.read_csv('/home/tri/llms-anomaly-detection/TSAD-NUS/Dataset/Univariate_data/145_UCR_Anomaly_Lab2Cmac011215EPG1_5000_17210_17260/145_UCR_Anomaly_Lab2Cmac011215EPG1_5000_17210_17260.txt', header=None, names=['value'])
df = df[:5000]
abnormal_descriptions = load_abnormal_descriptions('/home/tri/llms-anomaly-detection/TSAD-NUS/Dataset/Univariate_data/145_UCR_Anomaly_Lab2Cmac011215EPG1_5000_17210_17260/description_abnormal.txt')
normal_description = load_normal_descriptions('/home/tri/llms-anomaly-detection/TSAD-NUS/Dataset/Univariate_data/145_UCR_Anomaly_Lab2Cmac011215EPG1_5000_17210_17260/description_normal.txt')

# Initialize a column for descriptions with the normal description
df['description'] = normal_description

for desc in abnormal_descriptions:
    df.loc[df.index.isin(range(desc['start'], desc['end'] + 1)), 'description'] = desc['reason']

model = AutoModelForCausalLM.from_pretrained("facebook/opt-350m")
# model.to(device)
tokenizer = AutoTokenizer.from_pretrained("facebook/opt-350m")

def formatting_prompts_func(example):
    output_texts = []
    for i in range(len(example['instruction'])):
        text = f"### Question: {example['instruction'][i]}\n ### Answer: {example['output'][i]}"
        output_texts.append(text)
    return output_texts

def construct_general_example(df, normal_description):
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
                       "Given the values provided, classify the interval as 'normal'. Please provide your reasoning based on the values and their expected pattern.\n")
        
        # Append the constructed texts to their respective lists
        instructions.append(instruction)
        answers.append(f"The normal pattern is {normal_description}")

    return instructions, answers



instruction, outputs = construct_general_example(df, normal_description)


def formatting_prompts_func(example):
    output_texts = []
    for i in range(len(example['question'])):
        text = f"### Question: {example['question'][i]}\n ### Answer: {example['answer'][i]}"
        output_texts.append(text)
    return output_texts

dataframe = pd.DataFrame({'question': instruction, 'answer': outputs})

dataset = Dataset.from_pandas(dataframe)
trainer = SFTTrainer(
    model,
    train_dataset=dataset,
    formatting_func=formatting_prompts_func,
)

trainer.train()