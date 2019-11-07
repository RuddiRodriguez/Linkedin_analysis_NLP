#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 24 23:35:47 2019
In this module there are a set of functions desdicated to transform and prepared
corpus data for further analysis using mchaine learning tools. 
@author: ruddirodriguez
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
# import re
#from nltk.stem import PorterStemmer
import nltk
from nltk.corpus import stopwords
from stemming.porter2 import stem
#from textblob import Word
from nltk.stem import WordNetLemmatizer
from sklearn import model_selection, preprocessing



def training_model_with_split(data, test_size, features_col, labels_col):

    list_features = data[features_col].tolist()
    list_labels = data[labels_col].tolist()

    X_train, X_test, y_train, y_test = train_test_split(list_features, list_labels, test_size=test_size,
                                                        random_state=40)
    return X_train, X_test, y_train, y_test


def cleaning_text_regular_exp(df, text_field):
    #df[text_field] = df[text_field].str.replace(r"http\S+", "")
    df[text_field] = df[text_field].str.replace(r'[^\w\s]', " ")
    df[text_field] = df[text_field].str.replace(r":", "")
    df[text_field] = df[text_field].str.replace(
        r"[^A-Za-z0-9(),!?@\'\`\"\_\n]", " ")
    df[text_field] = df[text_field].str.replace(r"\d", " ")
    df[text_field] = df[text_field].str.lower()
    return df


def removing_stop_words(data):
    nltk.download("stopwords")
    stop = stopwords.words('english')
    data['token_no_stop_words'] = data['Description'].apply(
        lambda x: " ".join([item for item in x.split() if item not in stop]))
    return (data)


def steamming(data, column_name):
    #st = PorterStemmer()
    data['stem'] = data[column_name].apply(
        lambda x: " ".join([stem(word) for word in x.split()]))
    # data['stem'] = data['token_no_stop_words'].apply(lambda x: [st.stem(word) for
    # word in x])
    return data


def lemmatizing(data, colum_name):
    # st = PorterStemmer()
    nltk.download('wordnet')
    lemmatizer = WordNetLemmatizer()
    data['lemmatizing'] = data[colum_name].apply(
        lambda x: " ".join([lemmatizer.lemmatize(word) for word in x.split()]))
    # data['stem'] = data['token_no_stop_words'].apply(lambda x: [st.stem(word) for
    # word in x])
    return data


def data_categorization(data):
    data['Cat_level'] = data['Level']
    labels = data['Level'].astype('category').cat.categories.tolist()
    replace_map_comp = {'Level': {k: v for k, v in zip(
        labels, list(range(1, len(labels)+1)))}}
    data.replace(replace_map_comp, inplace=True)
    print(replace_map_comp)

    return data


def data_encoder(data):
    encoder = preprocessing.LabelEncoder()
    data_categor = encoder.fit_transform(data)

    return data_categor
