#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 26 00:53:08 2019

@author: ruddirodriguez
"""
import pandas as pd 
from module_transformers import FeatureSelector
# Create a function that
def drop_nan(df, col):
    # drop nan values
    return df[df[col] != 'Not Applicable']

# Create a function that
def data_categorization(df, col):
    df['Cat_level'] = df[col]
    labels = df[col].astype('category').cat.categories.tolist()
    replace_map_comp = {col: {k: v for k, v in zip(
        labels, list(range(1, len(labels)+1)))}}
    df.replace(replace_map_comp, inplace=True)
    #print(replace_map_comp)

    return df


training_data_name = 'demodata_training_full_v1.csv'
training_data = pd.read_csv('Data/'+training_data_name)
# Create a pipeline that applies the mean_age_by_group function
cleaned_data = (training_data.pipe(drop_nan, col='Level')
   # then applies the uppercase column name function
   .pipe(data_categorization, col='Level')
)

test = FeatureSelector ('Level')
dd = test.transform (cleaned_data)



