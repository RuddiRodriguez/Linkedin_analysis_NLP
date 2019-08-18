#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Aug 17 02:51:08 2019

@author: ruddirodriguez
"""

# !/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Aug  3 03:06:25 2019

@author: ruddirodriguez
"""

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline

# from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report
import module_data as md
import module_model as mo
import module_plot as plot
from module_print import print_summary


# Functions


# def getting_working_data (data):


def preparing_data(data):
    data = data[['Level', 'Type_of_Job', 'Description']]
    data = data[data.Level != 'Not Applicable']
    data_no_regular_expression = md.cleaning_text_regular_exp(
        data, 'Description')
    data_no_stop_words = md.removing_stop_words(data_no_regular_expression)
    # data_stemming = steamming(data_no_stop_words)
    data_lemmatizing = md.lemmatizing(data_no_stop_words, "Description")
    data_categorical = md.data_categorization(data_lemmatizing)

    return data_categorical


def ML_analysis(X_train, X_test, y_train, y_test, data):
    X_train_counts, count_vectorizer = mo.vectorizer(
        X_train, (2, 2))

    # transforming testing data into document-term matrix
    X_test_counts = count_vectorizer.transform(
        X_test)

    pipeline = Pipeline([('clf', LogisticRegression(solver='saga',multi_class='multinomial',class_weight='balanced'))])

    parameters = {'clf__penalty': ['l1', 'l2'],
                  'clf__C': [0.001, .009, 0.01, .09, 1,2, 5, 10, 25, 30, 40]}
    #    scorers = {
    #    'precision_score': make_scorer(precision_score),
    #    'recall_score': make_scorer(recall_score),
    #    'accuracy_score': make_scorer(accuracy_score)}

    scoring = ['precision_macro', 'recall_macro', 'f1_macro', 'balanced_accuracy']

    grid_search = GridSearchCV(pipeline, parameters, cv=5,
                               n_jobs=-1, verbose=1, scoring=scoring, refit='recall_macro')
    #
    grid_search.fit(X_train_counts, y_train)

    best_parameters = grid_search.best_params_
    print(best_parameters)
    print(grid_search.best_score_)
    predicted_level = grid_search.predict(X_test_counts)
    print(grid_search.best_estimator_)
    print_summary(y_test, predicted_level,
                  data, "Cat_level")
    plot.plot_confusion_matrix(y_test, predicted_level, classes=data.groupby('Cat_level').count().index,
                               title='Confusion matrix, without normalization')


real_data_name = 'demodata.csv'
training_data_name = 'demodata_training_full_v1.csv'

training_data = pd.read_csv(
    '/Users/ruddirodriguez/Dropbox/Machine_Learning/NLP/' + training_data_name)
Real_data = pd.read_csv(
    '/Users/ruddirodriguez/Dropbox/Machine_Learning/NLP/' + real_data_name)
real_data_clean = preparing_data(Real_data)
training_data_clean = preparing_data(training_data)

y_train = training_data_clean["Level"].tolist()
y_test = real_data_clean["Level"].tolist()

ML_analysis(training_data_clean['lemmatizing'].tolist(), real_data_clean['lemmatizing'].tolist(
), training_data_clean['Level'].tolist(), real_data_clean['Level'].tolist(), training_data_clean)
