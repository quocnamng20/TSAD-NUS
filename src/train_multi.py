import os
import pandas as pd
import numpy as np
import torch

from transformers import GPT2Tokenizer, GPT2LMHeadModel, TrainingArguments, Trainer, GPT2ForSequenceClassification, pipeline, AutoModelForCausalLM, AutoTokenizer, AutoModelForSeq2SeqLM, AdamW, get_linear_schedule_with_warmup
from torch.utils.data import DataLoader
from datasets import load_metric, load_dataset, Dataset
from sklearn.model_selection import train_test_split
from trl import SFTTrainer, DataCollatorForCompletionOnlyLM

from dataloader import load_abnormal_descriptions, load_normal_descriptions, match_anomalies_with_descriptions, construct_multi_general_example

# Load your data
df = pd.read_csv('/home/tri/llms-anomaly-detection/TSAD-NUS/Dataset/Multivariate_data/Gen_1/gen_2.csv')
abnormal_descriptions = load_abnormal_descriptions('/home/tri/llms-anomaly-detection/TSAD-NUS/Dataset/Multivariate_data/Gen_1/description_abnormal.txt')
normal_description = load_normal_descriptions('/home/tri/llms-anomaly-detection/TSAD-NUS/Dataset/Multivariate_data/Gen_1/description_normal.txt')

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



instruction, outputs = construct_multi_general_example(df, normal_description)

response_template = " ### Answer:"

def formatting_prompts_func(example):
    output_texts = []
    for i in range(len(example['question'])):
        text = f"### Question: {example['question'][i]}\n ### Answer: {example['answer'][i]}"
        output_texts.append(text)
    return output_texts

dataframe = pd.DataFrame({'question': instruction, 'answer': outputs})
dataframe.to_csv('/home/tri/llms-anomaly-detection/TSAD-NUS/Dataset/training.csv', sep='\t')

dataset = Dataset.from_pandas(dataframe)
train_dataset = Dataset.from_pandas(dataframe)
# Define training arguments

training_args = TrainingArguments(
    output_dir='./results',          # The output directory where the model predictions and checkpoints will be written.
    num_train_epochs=3,              # Total number of training epochs
    per_device_train_batch_size=8,   # Batch size per device during training
    per_device_eval_batch_size=8,    # Batch size for evaluation
    warmup_steps=500,                # Number of warmup steps for learning rate scheduler
    weight_decay=0.01,               # Strength of weight decay
    logging_dir='./logs',            # Directory for storing logs
    logging_steps=10,
    save_strategy="epoch",           # Save strategy to save a model every epoch
    args=training_args,
)

trainer = SFTTrainer(
    model,
    train_dataset=train_dataset,
    formatting_func=formatting_prompts_func,
)

model_path = "./model_checkpoint"
model.save_pretrained(model_path)
tokenizer.save_pretrained(model_path)

trainer.train()