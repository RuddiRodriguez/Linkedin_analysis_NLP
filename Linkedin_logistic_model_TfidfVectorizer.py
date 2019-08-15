#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 12 23:28:52 2019

@author: ruddirodriguez
"""


import pandas as pd
from sklearn.linear_model import LogisticRegression
import module_data as md
import module_model as mo
import module_plot as plot
import nltk
from nltk.corpus import stopwords
from module_print import print_summary

# function to cleaning and prepare the data


def preparing_data(data):
    data = data[['Level', 'Type_of_Job', 'Description']]
    data = data[data.Level != 'Not Applicable']
    return (data)


def ML_analysis(X_train, X_test, y_train, y_test, data):

    # Encoding data to apply machine learning tools
    y_train_encoder = md.data_encoder(y_train)
    y_test_encoder = md.data_encoder(y_test)

    # setting stop words
    nltk.download("stopwords")
    stop = stopwords.words('english')

    # vectorization
    tfidf_vect = mo.vectorizer_Tfid(training_data_clean, "Description", stop_words=stop,
                                    token_pattern=r'\w{1,}', ngram_range=(1, 2), max_features=1000)
    X_train_counts = tfidf_vect.transform(X_train)
    X_test_counts = tfidf_vect.transform(X_test)

    logreg = LogisticRegression()  # C=30.0, class_weight='balanced', solver='newton-cg',
    # multi_class='multinomial', n_jobs=-1, random_state=40)

    predicted_level = mo.train_model(
        logreg, X_train_counts, y_train_encoder, X_test_counts)

    print_summary(y_test_encoder, predicted_level,
                  training_data_clean, "Level")

    plot.plot_confusion_matrix(y_test_encoder, predicted_level,
                               classes=data.groupby(
                                   'Level').count().index,
                               title='Confusion matrix, without normalization')
#
#


training_data_name = 'demodata.csv'

training_data = pd.read_csv(
    '/Users/ruddirodriguez/Dropbox/Machine_Learning/NLP/'+training_data_name)

# spliting data

training_data_clean = preparing_data(training_data)
X_train, X_test, y_train, y_test = md.training_model_with_split(
    training_data_clean, 0.2, "Description", "Level")

ML_analysis(X_train, X_test, y_train, y_test, training_data_clean)


real_data_name = 'demodatareal2.csv'
Real_data = pd.read_csv(
    '/Users/ruddirodriguez/Dropbox/Machine_Learning/NLP/'+real_data_name)
real_data_clean = preparing_data(Real_data)

ML_analysis(training_data_clean['Description'].tolist(), real_data_clean['Description'].tolist(
), training_data_clean['Level'].tolist(), real_data_clean['Level'].tolist(), training_data_clean)
