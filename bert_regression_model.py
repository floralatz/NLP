# https://jalammar.github.io/a-visual-guide-to-using-bert-for-the-first-time/


import numpy as np
import pandas as pd
import torch
import json
import pickle
import transformers as ppb # pytorch transformers
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.dummy import DummyClassifier

def save_to_pickle(dataset, fileloc):
    """
    Saves dataset to a pickle to specified location
    """
    filename = fileloc + ".pickle"
    with open(filename, 'wb') as f:
        pickle.dump(dataset, f)

def load_from_pickle(fileloc):
    """
    Loads dataset from a pickle
    """
    filename = fileloc + ".pickle"
    with open(filename, 'rb') as f:
        dataset = pickle.load(f)
    return dataset

def create_dataset(which_set):
    """
    Creates the dataset from .txt files
    :param which_set: train, test or dev
    :return: dataframe with texts and labels
    """

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
            one_hot_labels.append(0)  # 0 = no propaganda
        else:
            one_hot_labels.append(1)  # 1 = propaganda

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


def get_features_and_labels_with_bert(df):
    """
    creates features and labels using the DISTILBERT Model
    :param df: dataframe with texts and labels
    :return: features & labels
    """

    model_class, tokenizer_class, pretrained_weights = (ppb.DistilBertModel, ppb.DistilBertTokenizer, 'distilbert-base-uncased')

    # Load pretrained model/tokenizer
    tokenizer = tokenizer_class.from_pretrained(pretrained_weights)
    model = model_class.from_pretrained(pretrained_weights)

    # This turns every sentence into the list of ids
    tokenized = df['text'].apply((lambda x: tokenizer.encode(x, add_special_tokens=True)))

    # Padding
    max_len = 0
    for i in tokenized.values:
        if len(i) > max_len:
            max_len = len(i)
    padded = np.array([i + [0]*(max_len-len(i)) for i in tokenized.values])

    # Masking
    attention_mask = np.where(padded != 0, 1, 0)

    # Model
    input_ids = torch.tensor(padded)
    attention_mask = torch.tensor(attention_mask)

    with torch.no_grad():
        last_hidden_states = model(input_ids, attention_mask=attention_mask)

    features = last_hidden_states[0][:, 0, :].numpy()
    labels = df['label']

    return features, labels


if __name__ == "__main__":

    # PARAMETERS
    saveto = False
    loadfrom = False
    savemodel = False

    # Create datasets
    df_train = create_dataset('train')
    df_test = create_dataset('test')

    # Use Bert for features and labels
    train_features, train_labels = get_features_and_labels_with_bert(df_train)
    test_features, test_labels = get_features_and_labels_with_bert(df_test)

    # Save features and labels to pickle
    if saveto:
        save_to_pickle(train_features, 'train_features')
        save_to_pickle(train_labels, 'train_labels')
        save_to_pickle(test_features, 'test_features')
        save_to_pickle(test_labels, 'test_labels')

    # Load features and labels from pickle
    if loadfrom:
        train_features = load_from_pickle('train_features')
        train_labels = load_from_pickle('train_labels')
        test_features = load_from_pickle('test_features')
        test_labels = load_from_pickle('test_labels')


    ######## Logistic Regression #########
    # Fit Model
    lr_clf = LogisticRegression(solver='lbfgs', max_iter=200)
    lr_clf.fit(train_features, train_labels)

    # Save Model
    if savemodel:
        save_to_pickle(lr_clf, "logistic_regression_model")

    # Evaluation
    score = lr_clf.score(test_features, test_labels)
    print("Score is", score)

    # Comparison
    clf = DummyClassifier()

    scores = cross_val_score(clf, train_features, train_labels)
    print("Dummy classifier score: %0.3f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
