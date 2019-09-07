#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 26 00:53:08 2019
#testd = CleaningTextRegularExp ('Description')
#dd = testd.transform ( )
#
#
#testr = removing_stop_words ('Description')
#ddd = testr.transform (dd)
#
#test = FeatureSelector ('Description')
#dddd = test.transform (ddd)
@author: ruddirodriguez
"""
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression

from sklearn.pipeline import Pipeline
from module_model import ML_analysis_split, ML_analysis_separated_data


# Create a function that
def drop_nan(df, col):
    # drop nan values
    return df[df[col] != 'Not Applicable']


# Create a function that
def data_categorization(df, col):
    df['Cat_level'] = df[col]
    labels = df[col].astype('category').cat.categories.tolist()
    replace_map_comp = {col: {k: v for k, v in zip(
        labels, list(range(1, len(labels) + 1)))}}
    df.replace(replace_map_comp, inplace=True)
    # print(replace_map_comp)

    return df


def loading_data():
    training_data_name = 'demodata_training_full_v1.csv'
    training_data = pd.read_csv('Data/' + training_data_name)
    # Create a pandas pipeline that prepare the data
    cleaned_data = (training_data.pipe(drop_nan, col='Level')
                    # then applies the categorization
                    .pipe(data_categorization, col='Level')
                    )
    # ML_analysis_split(cleaned_data, "Level",LogisticRegression,"Cat_level")

    test_data_name = 'demodata.csv'
    test_data = pd.read_csv('Data/' + test_data_name)
    cleaned_data_test = (test_data.pipe(drop_nan, col='Level')
                         # then applies the categorization
                         .pipe(data_categorization, col='Level')
                         )

    ML_analysis_separated_data(cleaned_data, cleaned_data_test, "Level", LogisticRegression, "Cat_level")


if __name__ == '__main__':
    loading_data()
