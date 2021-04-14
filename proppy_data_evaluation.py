
import pickle
from nltk.tokenize import sent_tokenize
import numpy as np
import transformers as ppb # pytorch transformers
import torch
import pandas as pd
from bert_regression_model import get_features_and_labels_with_bert


def load_from_pickle(fileloc):
    """
    Loads dataset from a pickle
    """
    filename = fileloc + ".pickle"
    with open(filename, 'rb') as f:
        dataset = pickle.load(f)
    return dataset


# load data from pickle
filename = "proppy_data_evaluation" + ".pickle"
with open(filename, 'rb') as f:
    df = pickle.load(f)

# create df for every article
for index, article in enumerate(df['text']):
    print("########  INDEX:", index)
    tokenized_sentences = []
    current_label = df['labels'][index]
    tokenized_sentences.append(sent_tokenize(article))
    data = {'text': sent_tokenize(article), 'label': current_label}
    article_df = pd.DataFrame(data)

    # Get features from BERT
    features, labels = get_features_and_labels_with_bert(article_df)

    # Load logistic regression model
    lr_model = load_from_pickle("logistic_regression_model")

    # predict labels
    pred = lr_model.predict(features)
    print("Prediction of LR_model is", pred)
    print("Correct label is ", current_label)

    # combine Label
    one_count = np.count_nonzero(pred == 1)
    zero_count = np.count_nonzero(pred == 0)
    # assumption: if the majority of the article is propaganda, the entire article is propaganda
    # alternative: if one sentence is propaganda, the entire article is propaganda
    if one_count >= zero_count:
        predicted_label = 1
    else:
        predicted_label = 0
    print("Predicted label is ", predicted_label)

    # check label
    if predicted_label == current_label:
        print("Correct Label assigned")
