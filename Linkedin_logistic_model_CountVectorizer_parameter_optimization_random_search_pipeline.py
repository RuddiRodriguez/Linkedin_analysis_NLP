#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Aug 18 22:20:36 2019

@author: ruddirodriguez
"""
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import RandomizedSearchCV
from sklearn.pipeline import Pipeline

import module_data as md
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
    # transforming testing data into document-term matrix
    # X_test_counts = count_vectorizer.transform(
    #   X_test)

    # #############################################################################
    # Define a pipeline combining a text feature extractor with a simple
    # classifier
    pipeline = Pipeline([('vectorized', CountVectorizer()), ('clf',
                                                             LogisticRegression(solver='saga',
                                                                                multi_class='multinomial',
                                                                                class_weight='balanced'))])

    param_grid = {'vectorized__ngram_range': [(1, 1), (1, 2), (1, 3), (1, 4), (1, 5), (2, 3)],
                  'vectorized__max_features': [300, 1000, 5000, 8000, 9000],
                  'clf__penalty': ['l1', 'l2'],
                  'clf__C': [1, 2, 5, 10, 25, 30, 40]}

    #    scorers = {
    #    'precision_score': make_scorer(precision_score),
    #    'recall_score': make_scorer(recall_score),
    #    'accuracy_score': make_scorer(accuracy_score)}

    scoring = ['precision_macro', 'recall_macro', 'f1_macro',
               'balanced_accuracy']

    grid_search = RandomizedSearchCV(pipeline, param_grid, cv=10, n_iter=10,
                                     n_jobs=-1, verbose=1, scoring=scoring, refit='recall_macro')
    #
    grid_search.fit(X_train, y_train)
    # grid_search.best_score_

    print(grid_search.best_params_)
    # print(grid_search.best_score_)
    predicted_level = grid_search.predict(X_test)
    # print(grid_search.best_estimator_)
    print_summary(y_test, predicted_level,
                  data, "Cat_level")
    plot.plot_confusion_matrix(
        y_test, predicted_level, classes=data.groupby(
            'Cat_level').count().index,
        title='Confusion matrix, without normalization')

    best_parameters = grid_search.best_estimator_.get_params()
    print('Best Parameters are')
    for param_name in sorted(param_grid.keys()):
        print("\t%s: %r" % (param_name, best_parameters[param_name]))

    return predicted_level, grid_search


real_data_name = 'demodata.csv'
training_data_name = 'demodata_training_full_v1.csv'

training_data = pd.read_csv(
    '/Users/ruddirodriguez/Dropbox/Machine_Learning/NLP/' + training_data_name)
Real_data = pd.read_csv(
    '/Users/ruddirodriguez/Dropbox/Machine_Learning/NLP/' + real_data_name)
training_data_clean = preparing_data(training_data)
real_data_clean = preparing_data(Real_data)

y_train = training_data_clean["Level"].tolist()
# y_test = real_data_clean["Level"].tolist()

(predictions, model) = ML_analysis(training_data_clean['lemmatizing'].tolist(),
                                   real_data_clean['lemmatizing'].tolist(),
                                   training_data_clean['Level'].tolist(),
                                   real_data_clean['Level'].tolist(), training_data_clean)
