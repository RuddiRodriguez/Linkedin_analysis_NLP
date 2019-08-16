#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 14 02:40:59 2019

@author: ruddirodriguez
"""

from sklearn import metrics

def print_summary(y_test, y_predicted,data,colum_name):
    
    print(metrics.classification_report(y_test, y_predicted, target_names=data.groupby(colum_name).count().index))