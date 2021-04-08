# https://jalammar.github.io/a-visual-guide-to-using-bert-for-the-first-time/


import numpy as np
import pandas as pd
import torch
import json
import transformers as ppb # pytorch transformers
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split


def create_dataset(which_set):
    if which_set == 'train':
        with open('2021_data/training_set_task1.txt') as json_file:
            data = json.load(json_file)
    elif which_set == 'test':
        with open('2021_data/test_set_task1.txt') as json_file:
            data = json.load(json_file)
    elif which_set == 'dev':
        with open('2021_data/dev_set_task1.txt') as json_file:
            data = json.load(json_file)

    # Create labels array
    labels = []
    for da in data:
        labels.append(da['labels'])

    # One hot encoding of labels
    one_hot_labels = []
    for label in labels:
        if len(label) == 0:
            one_hot_labels.append(0)
        else:
            one_hot_labels.append(1)

    # Create Text array
    texts = []
    for da in data:
        texts.append(da['text'])

    # Create Pandas Dataframe
    data = {'text': texts,
            'label': one_hot_labels
            }
    df = pd.DataFrame(data, columns = ['text', 'label'])

    return df

df = create_dataset('train')

model_class, tokenizer_class, pretrained_weights = (ppb.DistilBertModel, ppb.DistilBertTokenizer, 'distilbert-base-uncased')

# Load pretrained model/tokenizer
tokenizer = tokenizer_class.from_pretrained(pretrained_weights)
model = model_class.from_pretrained(pretrained_weights)
tokenized = df[0].apply((lambda x: tokenizer.encode(x, add_special_tokens=True)))