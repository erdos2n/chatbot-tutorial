import os
import json
import joblib
import glob
import datetime
import numpy as np
import pandas as pd

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB


def load_data(filepath, filename):
    full_path = os.path.join(filepath, filename)
    with open(full_path) as f:
        data = json.load(f)
    train_data = []
    for label, lst in data.items():
        for utterance in lst:
            row = [utterance, label]
            train_data.append(row)
    return np.array(train_data)


def get_bow_df(train_data, transformers_path):
    X = train_data[:, 0].tolist()
    bow = CountVectorizer()
    vecs = bow.fit_transform(X).toarray()
    if transformers_path:
        filename="bow.pkl"
        joblib.dump(bow, os.path.join(transformers_path, filename))
    vocab = bow.get_feature_names()
    vocab_df = pd.DataFrame(vecs, columns=vocab)
    vocab_df['target'] = train_data[:, 1]
    return vocab_df


def make_model(train_data, models_info_path=None, save_path=None, transformers_path=None):
    bow_df = get_bow_df(train_data, transformers_path=transformers_path)
    X = bow_df.drop(columns=['target'])
    y = bow_df['target']
    clf = MultinomialNB()
    clf.fit(X, y)
    if save_path:
        save_model(clf, save_path)
        store_model_info(clf, X, y, models_info_path)
    return clf


def save_model(clf, save_path):
    now = datetime.datetime.now()
    now = now.strftime("%y-%m-%dT%H:%M:%S.%f")
    file_name = f"multiNB_{now}.pkl"
    full_file_name = os.path.join(save_path, file_name)
    joblib.dump(clf, full_file_name)
    return None


def store_model_info(clf, X, y, models_info_path):
    train_score = clf.score(X, y)
    file_name = get_latest_model()
    with open(os.path.join(models_info_path, "models_info.csv"), "a+") as f:
        f.write(f"{train_score}, {file_name}\n")
    return None


def get_latest_model():
    files = glob.glob("../models/*.pkl")
    latest_model = sorted(files, reverse=True)[0]
    return latest_model


if __name__=="__main__":
    curdir = os.getcwd()
    data_dir = os.path.join(curdir, "../data")
    training_data_filename = "training_data.json"
    model_path = "../models"
    models_info_path = "../models_info"
    transformers_path = "../transformers"
    train_data = load_data(filepath=data_dir, filename=training_data_filename)
    clf = make_model(train_data,
                     models_info_path=models_info_path,
                     save_path=model_path,
                     transformers_path=transformers_path)
