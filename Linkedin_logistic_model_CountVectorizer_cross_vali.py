#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 16 22:14:42 2019

@author: ruddirodriguez
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Aug  3 15:42:26 2019

@author: ruddirodriguez
"""

import pandas as pd
from sklearn.linear_model import LogisticRegression
import module_data as md
import module_model as mo
from sklearn import metrics
from sklearn.model_selection import cross_val_predict
from sklearn.model_selection import cross_val_score
# Functions


# def getting_working_data (data):


def preparing_data(data):
    data = data[['Level', 'Type_of_Job', 'Description']]
    data = data[data.Level != 'Not Applicable']
    data_no_regular_expression = md.cleaning_text_regular_exp(
        data, 'Description')
    data_no_stop_words = md.removing_stop_words(data_no_regular_expression)
    #data_stemming = steamming(data_no_stop_words)
    data_lemmatizing = md.lemmatizing(
        data_no_stop_words, "token_no_stop_words")
    data_categorical = md.data_categorization(data_lemmatizing)

    return (data_categorical)


def ML_analysis(X_train, y_train, data, colum_name):

    X_train_counts, count_vectorizer = mo.vectorizer(
        X_train, (1, 1))

    logreg = LogisticRegression(
        )  # C=30.0, class_weight='balanced', solver='newton-cg',
    # multi_class='multinomial', n_jobs=-1, random_state=40)
    #ist_features = X_train_counts
    #list_labels = data[colum_name].tolist()
    predicted = cross_val_predict(logreg, X_train_counts, y_train, cv=3)
    print(metrics.accuracy_score(y_train, predicted))
    print(metrics.classification_report(y_train, predicted))


training_data_name = 'demodata.csv'

training_data = pd.read_csv(
    '/Users/ruddirodriguez/Dropbox/Machine_Learning/NLP/'+training_data_name)

training_data_clean = preparing_data(training_data)
X_train = training_data_clean['lemmatizing']
y_train = training_data_clean['Level'].tolist()
ML_analysis(X_train, y_train, training_data_clean, 'Levels')
