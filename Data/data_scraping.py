#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 13 22:31:15 2019
This script was made to scrap in a given number of linkedin pages. The first 
block of code get the idnumber of the jobs from the preview information 
The second block of code goes trought each job getting the description of the job
@author: ruddirodriguez
"""

from bs4 import BeautifulSoup
import pandas as pd
import requests



# This piece of code get the idnumber of the jobs 
idnumber = []
for i in range(10, 500, 10):  # cycle through n number of pages of Linkedin job resources
    page = 'https://www.linkedin.com/jobs/search/?keywords=data+science&start=' + \
        str(i)
    web_result = requests.get(page).text
    soup = BeautifulSoup(web_result, "lxml")
    idnumber.extend([item['data-id']
                     for item in soup.find_all(attrs={"data-id": True})])
    
#checking only numerical id    
idnumber = [ x for x in idnumber if x.isdigit() ]    


# This piece of code get the description of the jobs and create a dataframe
work_level = []
description = []
number = 0
data = pd.DataFrame(columns=['Level', 'Type_of_Job', 'Description'])
for i in idnumber:  # cycle through of the jobs full information
    number += 1
    page = ('https://www.linkedin.com/jobs/view/'+str(i)+'/')
    web_result = requests.get(page).text
    soup = BeautifulSoup(web_result, "lxml")
    description = [b.text for b in soup.findAll(
        'div', {'class': 'description__text description__text--rich'})]
    work_level = [a.text for a in soup.findAll(
        'span', {'class': 'job-criteria__text job-criteria__text--criteria'})]
    work_level = work_level[0:2]
    work_level.extend(description)
    data.loc[number] = work_level


data.to_csv(
    r'/Path_to_save/name.csv ')


