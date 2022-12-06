#!/usr/bin/env python3
import pandas as pd
from transformers import AutoTokenizer
from datasets import load_dataset, Dataset, DatasetDict

tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")

df = pd.read_csv("data/data/uci-news-aggregator.csv")
df = df.dropna()
dataset = Dataset.from_pandas(df)

#data = load_dataset("csv", data_files="data/uci-news-rand-reduced.csv")
#dataset = data['train']
from datasets import ClassLabel
dataset = dataset.remove_columns(['ID', 'URL', 'PUBLISHER', 'STORY', 'HOSTNAME', 'TIMESTAMP'])
new_features = dataset.features.copy()
new_features["CATEGORY"] = ClassLabel(names=["b", "t", "m", "e"])
dataset = dataset.cast(new_features)
dataset = dataset.rename_column("CATEGORY", "label")
#data['train'] = dataset

def preprocess_function(examples):
    return tokenizer(examples["TITLE"], truncation=True, padding=True)


tokenized_data = dataset.map(preprocess_function, batched=True)
tokenized_data = tokenized_data.remove_columns(['TITLE'])
datasetdict = DatasetDict()
datasetdict['train'], datasetdict['test'] = tokenized_data.train_test_split(.1).values()


from transformers import DataCollatorWithPadding

data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
import evaluate

accuracy = evaluate.load("accuracy")

import numpy as np


def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    return accuracy.compute(predictions=predictions, references=labels)


id2label = {0: "b", 1: "t", 2: "m", 3: "e"}
label2id = {"b": 0, "t": 1, "m": 2, "e": 3}

from transformers import AutoModelForSequenceClassification, TrainingArguments, Trainer

model = AutoModelForSequenceClassification.from_pretrained(
    "distilbert-base-uncased", num_labels=len(id2label), id2label=id2label, label2id=label2id
)

training_args = TrainingArguments(
    output_dir="my_awesome_model",
    learning_rate=2e-5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=10,
    weight_decay=0.01,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=datasetdict["train"],
    eval_dataset=datasetdict["test"],
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)

trainer.train()
