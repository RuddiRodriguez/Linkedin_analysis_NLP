#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 26 00:53:08 2019

@author: ruddirodriguez
"""
import pandas as pd 
from module_transformers import FeatureSelector, cleaning_text_regular_exp, removing_stop_words
from sklearn.pipeline import FeatureUnion, Pipeline 
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from module_print import print_summary
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



#testd = cleaning_text_regular_exp ('Description')
#dd = testd.transform (cleaned_data)
#
#
#testr = removing_stop_words ('Description')
#ddd = testr.transform (dd)
#
#test = FeatureSelector ('Description')
#dddd = test.transform (ddd)

full_pipeline = Pipeline( steps = [ ( 'pre_regular_exp', cleaning_text_regular_exp ('Description') ),
                                    ( 'pre_stop_words', removing_stop_words ('Description')), 
                                    ( 'Pre_selector', FeatureSelector ('Description')),
                                    ( 'vectorized', CountVectorizer()),] )
    
    
#Leave it as a dataframe becuase our pipeline is called on a 
#pandas dataframe to extract the appropriate columns, remember?
X = cleaned_data.drop('Level',axis=1)
#You can covert the target variable to numpy 
y = cleaned_data['Level'].values 

X_train, X_test, y_train, y_test = train_test_split( X, y , test_size = 0.2 , random_state = 42 )

#The full pipeline as a step in another pipeline with an estimator as the final step
full_pipeline_m = Pipeline( steps = [ ( 'full_pipeline', full_pipeline),
                                  
                                  ( 'model', LogisticRegression() ) ] )

#Can call fit on it just like any other pipeline
full_pipeline_m.fit( X_train, y_train )  
#Can predict with it like any other pipeline
y_pred = full_pipeline_m.predict( X_test ) 
print_summary(y_test, y_pred,
                  cleaned_data, "Cat_level")
  
