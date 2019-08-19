#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug  1 21:41:08 2019

@author: ruddirodriguez
"""

import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
#from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report
import module_data as md
import module_model as mo
import module_plot as plot
import matplotlib.pyplot as plt
from module_print import print_summary

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


real_data_name = 'demodata.csv'
training_data_name = 'demodata_training_full_v1.csv'

training_data = pd.read_csv(
    '/Users/ruddirodriguez/Dropbox/Machine_Learning/NLP/'+training_data_name)
Real_data = pd.read_csv(
    '/Users/ruddirodriguez/Dropbox/Machine_Learning/NLP/'+real_data_name)
real_data_clean = preparing_data(Real_data)
training_data_clean = preparing_data(training_data)

#real_data_clean = real_data_clean[real_data_clean['Level'] ==5]
#training_data_clean = training_data_clean[(
#    training_data_clean['Level'] == 5) | (training_data_clean['Level'] == 2)]


#fig = plt.figure(figsize=(8, 6))
#plot.number_of_levels(real_data_clean)
#
#fig = plt.figure(figsize=(8, 6))
#plot.number_of_levels(training_data_clean)

X_train_counts, count_vectorizer = mo.vectorizer(
    training_data_clean['lemmatizing'].tolist(),(1,1))

# transforming testing data into document-term matrix
X_test_counts = count_vectorizer.transform(
    real_data_clean['lemmatizing'].tolist())

y_train = training_data_clean["Level"].tolist()
y_test = real_data_clean["Level"].tolist()


NB = MultinomialNB(alpha=0.5) 

predicted_level = mo.train_model(
    NB, X_train_counts, y_train, X_test_counts)

print_summary(y_test, predicted_level,
                  training_data_clean, "Cat_level")


accuracy, precision, recall, harmonic_mean = mo.get_metrics(
    y_test, y_predicted_counts)


#print("accuracy = %.3f, precision = %.3f, recall = %.3f, f1 = %.3f" % (accuracy, precision, recall, harmonic_mean))

evaluation_list = {'Accuracy': accuracy, 'Precision': precision,
                   'Recall': recall, 'Harmonic mean': harmonic_mean}

print(evaluation_list)

plot.plot_confusion_matrix(y_test, y_predicted_counts,
                           classes=real_data_clean.groupby(
                               'Cat_level').count().index,
                           title='Confusion matrix, without normalization')


