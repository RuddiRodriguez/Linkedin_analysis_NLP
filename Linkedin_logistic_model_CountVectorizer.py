#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 24 12:56:05 2019
 
@author: ruddirodriguez
 
This script implement a logistic model , on a data  collected from linkedin. 
The aim of the model is to predict the seniority level of a given job 
"""
import pandas as pd
import numpy as np
 
from sklearn.linear_model import LogisticRegression
#from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report
import module_data as md
import module_model as mo
import module_plot as plot
from module_print import print_summary

# Functions
def preparing_data(data):
    data = data[['Level', 'Type_of_Job', 'Description']]
    data = data[data.Level != 'Not Applicable']
    data = md.cleaning_text_regular_exp(
        data, 'Description')
    data = md.removing_stop_words(data)
    #data_stemming = steamming(data_no_stop_words)
    data = md.lemmatizing(data,"token_no_stop_words")
    data_cleaned = md.data_categorization(data)
 
    return (data_cleaned)
  
def ML_analysis(X_train, X_test, y_train, y_test, data):
 
    X_train_counts, count_vectorizer = mo.vectorizer(
        X_train,(1,2))
     
    X_test_counts = count_vectorizer.transform(X_test)
     
    logreg = LogisticRegression(C=1)  # C=30.0, class_weight='balanced', solver='newton-cg',
    # multi_class='multinomial', n_jobs=-1, random_state=40)
     
    predicted_level = mo.train_model(
        logreg, X_train_counts, y_train, X_test_counts)
     
    print_summary(y_test, predicted_level,
                  data, "Cat_level")
 
    plot.plot_confusion_matrix(y_test, predicted_level,
                               classes=data.groupby(
                                   'Cat_level').count().index,
                               title='Confusion matrix, without normalization')
 
# Load the data 
training_data_name = 'demodata_training_full_v1.csv'
 
training_data = pd.read_csv(training_data_name)
 
# Prepare the data 
training_data_clean = preparing_data(training_data)
X_train, X_test, y_train, y_test = md.training_model_with_split(
    training_data_clean, 0.2, "lemmatizing", "Level")
 
ML_analysis(X_train, X_test, y_train, y_test, training_data_clean)
 
 
real_data_name = 'demodata.csv'
Real_data = pd.read_csv(real_data_name)
real_data_clean = preparing_data(Real_data)

ML_analysis(training_data_clean['lemmatizing'].tolist(), real_data_clean['lemmatizing'].tolist(
), np.c_[training_data_clean['Level']], np.c_[real_data_clean['Level']], training_data_clean) 
 

#merge_data=training_data.append(Real_data)
#merge_data_clean = preparing_data(merge_data)
#X_train, X_test, y_train, y_test = md.training_model_with_split(
#    merge_data_clean, 0.3, "lemmatizing", "Level")
#    
#ML_analysis(X_train, X_test, y_train, y_test, merge_data_clean) 
#accuracy, precision, recall, harmonic_mean = mo.get_metrics(
#    y_test, y_predicted_counts)
#
#
##print("accuracy = %.3f, precision = %.3f, recall = %.3f, f1 = %.3f" % (accuracy, precision, recall, harmonic_mean))
#
#evaluation_list = {'Accuracy': accuracy, 'Precision': precision,
#                   'Recall': recall, 'Harmonic mean': harmonic_mean}
#
#print(evaluation_list)
#
#plot.plot_confusion_matrix(y_test, y_predicted_counts,
#                           classes=real_data_clean.groupby(
#                               'Cat_level').count().index,
#                           title='Confusion matrix, without normalization')
#                          
#print (real_data_clean.Level.value_counts ()/real_data_clean.shape[0])
