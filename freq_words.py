#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug  1 22:10:49 2019

@author: ruddirodriguez
"""

import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
import module_data as md
import module_model as mo
import module_plot as plot
import matplotlib.pyplot as plt
#% matplotlib inline
import numpy as np
from os import path
from PIL import Image
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator
# Functions


# def getting_working_data (data):


def preparing_data(data):
    data = data[['Level', 'Type_of_Job', 'Description']]
    data = data[data.Level != 'Not Applicable']
    data_no_regular_expression = md.cleaning_text_regular_exp(
        data, 'Description')
    data_no_stop_words = md.removing_stop_words(data_no_regular_expression)
    #data_stemming = steamming(data_no_stop_words)
    data_lemmatizing = md.lemmatizing(data_no_stop_words,"Description")
    data_categorical = md.data_categorization(data_lemmatizing)

    return (data_categorical)


def worcloud_generation (text,savepath):
    wordcloud = WordCloud(max_font_size=50, max_words=100).generate(text)

# Display the generated image:
    fig = plt.figure(figsize=(8, 6))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis("off")
    
    fig.savefig(savepath, bbox_inches='tight')
    plt.show()
    return plt
    

def multipleReplace(text, wordDict):
    """
    take a text and replace words that match the key in a dictionary
    with the associated value, return the changed text
    """
    for key in wordDict:
        text = text.replace(key, wordDict[key])
    return text

    


#real_data_name = 'demodatareal2.csv'
training_data_name = 'demodata_training_full_v1.csv'

training_data = pd.read_csv('Data/' + training_data_name)
#Real_data = pd.read_csv(
#    '/Users/ruddirodriguez/Dropbox/Machine_Learning/NLP/'+real_data_name)
#real_data_clean = preparing_data(Real_data)
training_data_clean = preparing_data(training_data)

#real_data_clean = real_data_clean[real_data_clean['Level'] ==5]
#training_data_clean = training_data_clean[(
#    training_data_clean['Level'] == 5) | (training_data_clean['Level'] == 2)]


#fig = plt.figure(figsize=(8, 6))
#plot.number_of_levels(real_data_clean,'Cat_level')
#
#fig = plt.figure(figsize=(8, 6))
#plot.number_of_levels(training_data_clean,'Cat_level')

#X_train_counts, count_vectorizer = mo.vectorizer(
#    training_data_clean['lemmatizing'].tolist(),(2,2))
#
## transforming testing data into document-term matrix
#X_test_counts = count_vectorizer.transform(
#    real_data_clean['lemmatizing'].tolist())
#
#y_train = training_data_clean["Level"].tolist()
#y_test = real_data_clean["Level"].tolist()


                      

#freqs = pd.DataFrame([(word, X_train_counts.getcol(idx).sum()) for word, idx in count_vectorizer.vocabulary_.items()],columns=['word','Freq']).sort_values(by='Freq',ascending=False)
#freqs.head()
#text = " ".join (item for item in training_data_clean['lemmatizing'].tolist())
#
#worcloud_generation (text)
## Create and generate a word cloud image:
#wordDict = {
#'data': '',
#'year': '',
#'science':''}
#text_copy = multipleReplace(text, wordDict)
#worcloud_generation (text_copy)

list_data_frames = []
cc=0
for i in training_data_clean['Cat_level'].value_counts().index:
    cc+=1
    data = training_data_clean[training_data_clean['Level'] == cc]
    X_train_counts, count_vectorizer = mo.vectorizer(
    data['lemmatizing'].tolist(),(1,2))
    freqss = pd.DataFrame([(word, X_train_counts.getcol(idx).sum(),i) for word, idx in count_vectorizer.vocabulary_.items()],columns=['word','Freq','Level']).sort_values(by='Freq',ascending=False)
    list_data_frames.append(freqss)
    text = " ".join (item for item in data['lemmatizing'].tolist())
    wordDict = {
            'data': '',
            'year': '',
            'science':''}
    savepath=("/Users/ruddirodriguez/Dropbox/Machine_Learning/Data_Science_day_UU/" + i + "." + "svg")
    text_copy = multipleReplace(text, wordDict)
    worcloud_generation (text_copy,savepath)
    
    


