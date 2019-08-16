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
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
#from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report
import module_data as md
import module_model as mo
import module_plot as plot
import matplotlib.pyplot as plt
import numpy as np
import module_croos_validation as cv
from sklearn.naive_bayes import MultinomialNB
# Functions


# def getting_working_data (data):


def preparing_data(data):
    data = data[['Level', 'Type_of_Job', 'Description']]
    data = data[data.Level != 'Not Applicable']
    data_no_regular_expression = md.cleaning_text_regular_exp(
        data, 'Description')
    data_no_stop_words = md.removing_stop_words(data_no_regular_expression)
    #data_stemming = steamming(data_no_stop_words)
    data_lemmatizing = md.lemmatizing(data_no_stop_words,"token_no_stop_words")
    data_categorical = md.data_categorization(data_lemmatizing)

    return (data_categorical)


training_data_name = 'demodata.csv'

training_data = pd.read_csv(
    '/Users/ruddirodriguez/Dropbox/Machine_Learning/NLP/'+training_data_name)

training_data_clean = preparing_data(training_data)



X_train_counts, count_vectorizer = mo.vectorizer(
    training_data_clean['lemmatizing'],(1,1))

# transforming testing data into document-term matrix
y_train =training_data_clean['Level'].tolist()


logreg = LogisticRegression()  # C=30.0, class_weight='balanced', solver='newton-cg',
# multi_class='multinomial', n_jobs=-1, random_state=40)

from sklearn import metrics
from sklearn.model_selection import cross_val_predict
from sklearn.model_selection import cross_val_score

list_features = X_train_counts
list_labels = training_data_clean['Level'].tolist()
predicted = cross_val_predict(logreg, X_train_counts, y_train, cv=10)
print (metrics.accuracy_score(y_train, predicted))
print (metrics.classification_report(y_train, predicted) )

                      
logreg = LogisticRegression()
# search for an optimal value of solver for LogisticRegresion
parameter_solver = ('newton-cg', 'lbfgs', 'liblinear')
# empty list to store scores
k_scores = []

# 1. we will loop through values of parameter_solver
for k in parameter_solver:
    # 2. run logisticregression
    logreg = LogisticRegression(solver=k)
    # 3. obtain cross_val_score for logisticregression
    scores = cross_val_score(logreg, X_train_counts, y_train, cv=10, scoring='accuracy')
    # 4. append mean of scores 
    k_scores.append(scores.mean())


print(k_scores)
