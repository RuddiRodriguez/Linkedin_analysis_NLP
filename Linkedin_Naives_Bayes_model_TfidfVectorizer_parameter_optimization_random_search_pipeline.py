#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 19 12:30:29 2019

@author: ruddirodriguez
"""

from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
from sklearn.linear_model import LogisticRegression
import module_data as md
import module_model as mo
import module_plot as plot
import nltk
from nltk.corpus import stopwords
from module_print import print_summary
from sklearn.metrics import recall_score
from sklearn.pipeline import Pipeline
from sklearn.model_selection import RandomizedSearchCV,GridSearchCV
from sklearn.naive_bayes import MultinomialNB
import numpy as np
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

    
    pipeline = Pipeline([('vectorizer',TfidfVectorizer(stop_words = stop,token_pattern=r'\w{1,}')),
                          ('clf', MultinomialNB(class_prior=None,fit_prior=False))])
    
   #solver='saga', multi_class='multinomial',
                                #class_weight='balanced'
                                
    param_grid={'vectorizer__ngram_range': [(1,1),(1,2) ,(2,2)],
                'vectorizer__max_features':[ 9000, 10000],
                'clf__alpha': np.linspace(0.5, 1.5, 6, 7)
                }  

    
    
    scoring = ['precision_macro', 'recall_macro', 'f1_macro',
               'balanced_accuracy']

    #grid_search = RandomizedSearchCV(pipeline, param_grid, cv=50, n_iter=30,
                                     #n_jobs=-1, verbose=1, scoring=scoring, refit='recall_macro')
    grid_search = GridSearchCV(pipeline, param_grid, cv=5,
                                     n_jobs=-1, verbose=1, scoring=scoring, refit='recall_macro')
    #
    #
    grid_search.fit(X_train, y_train_encoder)
    #    grid_search.best_score_
    best_parameters = grid_search.best_params_
    print(best_parameters)
    #    print(grid_search.best_score_)
    predicted_level = grid_search.predict(X_test)
    print(grid_search.best_estimator_)
    print_summary(y_test_encoder, predicted_level,
                  data, "Level")

#    predicted_level = mo.train_model(
#        logreg, X_train_counts, y_train_encoder, X_test_counts)
#
    print_summary(y_test_encoder, predicted_level,
                  training_data_clean, "Level")
#
    plot.plot_confusion_matrix(y_test_encoder, predicted_level,
                               classes=data.groupby(
                                   'Level').count().index,
                               title='Confusion matrix, without normalization')
#    plot.precision_number_training_data(training_data_clean,recall_score(y_test_encoder, predicted_level,average=None),'Level')

                               
    accuracy, precision, recall, harmonic_mean = mo.get_metrics(
    y_test_encoder, predicted_level)
#                               
    evaluation_list = {'Accuracy': accuracy, 'Precision': precision,
                   'Recall': recall, 'Harmonic mean': harmonic_mean}
#
#    print(evaluation_list)
    return predicted_level, grid_search
#
#




real_data_name = 'demodata.csv'
training_data_name = 'demodata_training_full_v1.csv'

training_data = pd.read_csv(
    '/Users/ruddirodriguez/Dropbox/Machine_Learning/NLP/' + training_data_name)
Real_data = pd.read_csv(
    '/Users/ruddirodriguez/Dropbox/Machine_Learning/NLP/' + real_data_name)
real_data_clean = preparing_data(Real_data)
training_data_clean = preparing_data(training_data)

#X_tain
#
#y_train = training_data_clean["Level"].tolist()
#y_test = real_data_clean["Level"].tolist()




(predictions,model) = ML_analysis(training_data_clean['Description'].tolist(),
            real_data_clean['Description'].tolist(),
            training_data_clean['Level'].tolist(),
            real_data_clean['Level'].tolist(), training_data_clean)


#merge_data=training_data.append(Real_data)
#merge_data_clean = preparing_data(merge_data)
#X_train, X_test, y_train, y_test = md.training_model_with_split(
#    merge_data_clean, 0.3, "Description", "Level")
#    
#ML_analysis(X_train, X_test, y_train, y_test, merge_data_clean) 

