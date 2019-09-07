#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 26 00:53:11 2019

@author: ruddirodriguez
"""

import nltk
from nltk.stem import WordNetLemmatizer
from sklearn.base import BaseEstimator, TransformerMixin


# Custom Transformer that extracts columns passed as argument to its constructor
class FeatureSelector(BaseEstimator, TransformerMixin):
    # Class Constructor
    def __init__(self, feature_names):
        self.feature_names = feature_names

        # Return self nothing else to do here

    def fit(self, X, y=None):
        return self

        # Method that describes what we need this transformer to do

    def transform(self, X, y=None):
        return X[
            self.feature_names].tolist()  # np.asarray(X[self.feature_names]).astype(str)#np.c_[X]#X[self.feature_names]#pd.DataFrame (X,columns= self.feature_names)


class CleaningTextRegularExp(BaseEstimator, TransformerMixin):
    # Class Constructor
    def __init__(self, col_name):
        self.col_name = col_name

        # Return self nothing else to do here

    def fit(self, X, y=None):
        return self

        # Method that describes what we need this transformer to do

    def transform(self, X, y=None):
        # df[text_field] = df[text_field].str.replace(r"http\S+", "")
        X[self.col_name] = X[self.col_name].str.replace(r'[^\w\s]', " ")
        X[self.col_name] = X[self.col_name].str.replace(r":", "")
        X[self.col_name] = X[self.col_name].str.replace(
            r"[^A-Za-z0-9(),!?@\'\`\"\_\n]", " ")
        X[self.col_name] = X[self.col_name].str.replace(r"\d", " ")
        X[self.col_name] = X[self.col_name].str.lower()

        return X


class removing_stop_words(BaseEstimator, TransformerMixin):
    # Class Constructor
    def __init__(self, col_name, language):
        self.col_name = col_name
        self.stopwords = set(nltk.corpus.stopwords.words(language))

    # Return self nothing else to do here
    def fit(self, X, y=None):
        return self

    # Method that describes what we need this transformer to do
    def transform(self, X, y=None):
        # nltk.download("stopwords")
        # stop = stopwords.words('english')
        X[self.col_name] = X[self.col_name].apply(
            lambda x: " ".join([item for item in x.split() if item not in self.stopwords]))
        return X


class LemmatizeWord(BaseEstimator, TransformerMixin):
    # Class Constructor
    def __init__(self, col_name):
        self.col_name = col_name
        self.lemmatizers = WordNetLemmatizer()

    # Return self nothing else to do here
    def fit(self, X, y=None):
        return self

    # Method that describes what we need this transformer to do
    def transform(self, X, y=None):
        # st = PorterStemmer()

        X[self.col_name] = X[self.col_name].apply(
            lambda x: " ".join([self.lemmatizers.lemmatize(item) for item in x.split()]))
        return X
