#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 24 23:52:19 2019
In this module there are a set of functions desdicated to fit a postprocessed
corpus
@author: ruddirodriguez
"""

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report
from sklearn.model_selection import KFold
from sklearn.feature_extraction.text import TfidfTransformer


def vectorizer(data, range_d):
    count_vectorizer = CountVectorizer(ngram_range=range_d)
    counts = count_vectorizer.fit_transform(data)  # .toarray()

    return counts, count_vectorizer


def vectorizer_Tfid(data):
    count_vectorizer = TfidfVectorizer()
    counts = count_vectorizer.fit_transform(data)

    return counts, count_vectorizer


def train_model(classifier, X_train, y_train, X_test, is_neural_net=False):
    # fit the training dataset on the classifier
    classifier.fit(X_train, y_train)
    # predict the labels on validation dataset
    predictions = classifier.predict(X_test)
    return predictions


def get_metrics(y_test, y_predicted):
    # true positives / (true positives+false positives)
    precision = precision_score(y_test, y_predicted, pos_label=None,
                                average='weighted')
    # true positives / (true positives + false negatives)
    recall = recall_score(y_test, y_predicted, pos_label=None,
                          average='weighted')

    # harmonic mean of precision and recall
    harmonic_mean = f1_score(
        y_test, y_predicted, pos_label=None, average='weighted')

#    # true positives + true negatives/ total
    accuracy = accuracy_score(y_test, y_predicted)
    return accuracy, precision, recall, harmonic_mean

    #print (accuracy, precision, recall, harmonic_mean)
