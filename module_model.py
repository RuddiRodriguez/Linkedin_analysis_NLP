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
import nltk
from nltk.corpus import stopwords
from sklearn.pipeline import Pipeline
from module_transformers import FeatureSelector, cleaning_text_regular_exp, removing_stop_words
from sklearn.model_selection import train_test_split
from module_print import print_summary


def vectorizer(data, range_d):
    count_vectorizer = CountVectorizer(ngram_range=range_d)
    counts = count_vectorizer.fit_transform(data)  # .toarray()

    return counts, count_vectorizer


def vectorizer_Tfid(data,colum_name ,preprocessor=None, tokenizer=None, stop_words=None, token_pattern='(?u)\b\w\w +\b', ngram_range=(1, 1), max_df=1.0, min_df=1, max_features=None, vocabulary=None):
    
    tfidf_vect = TfidfVectorizer(stop_words = stop_words)
    tfidf_vect.fit(data[colum_name])
    

    return tfidf_vect


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


def ML_analysis_split(cleaned_data, column_target,classifier,label=None):
    # Leave it as a dataframe because our pipeline is called on a
    # pandas dataframe to extract the appropriate columns, remember?

    X = cleaned_data.drop(column_target, axis=1)
    # You can covert the target variable to numpy
    y = cleaned_data[column_target].values

    full_pipeline = Pipeline(steps=[('pre_regular_exp', cleaning_text_regular_exp('Description')),
                                    ('pre_stop_words', removing_stop_words('Description')),
                                    ('Pre_selector', FeatureSelector('Description')),
                                    ('vectorized', CountVectorizer()), ])

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # The full pipeline as a step in another pipeline with an estimator as the final step
    full_pipeline_m = Pipeline(steps=[('full_pipeline', full_pipeline),

                                      ('model', classifier())])

    # Can call fit on it just like any other pipeline
    full_pipeline_m.fit(X_train, y_train)
    # Can predict with it like any other pipeline
    y_pred = full_pipeline_m.predict(X_test)
    print_summary(y_test, y_pred,
                  cleaned_data, label)


def ML_analysis_separated_data(training_cleaned_data, cleaned_test_data ,column_target,classifier,label=None):
    # Leave it as a dataframe because our pipeline is called on a
    # pandas dataframe to extract the appropriate columns, remember?

    X_train = training_cleaned_data.drop(column_target, axis=1)
    # You can covert the target variable to numpy
    y_train = training_cleaned_data[column_target].values
    
    X_test = cleaned_test_data.drop(column_target, axis=1)
    y_test = cleaned_test_data[column_target].values
    

    full_pipeline = Pipeline(steps=[('pre_regular_exp', cleaning_text_regular_exp('Description')),
                                    ('pre_stop_words', removing_stop_words('Description')),
                                    ('Pre_selector', FeatureSelector('Description')),
                                    ('vectorized', CountVectorizer()), ])

    

    # The full pipeline as a step in another pipeline with an estimator as the final step
    full_pipeline_m = Pipeline(steps=[('full_pipeline', full_pipeline),

                                      ('model', classifier())])

    # Can call fit on it just like any other pipeline
    full_pipeline_m.fit(X_train, y_train)
    # Can predict with it like any other pipeline
    y_pred = full_pipeline_m.predict(X_test)
    print_summary(y_test, y_pred,
                  training_cleaned_data, label)
