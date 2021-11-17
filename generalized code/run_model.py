# -*- coding: utf-8 -*-
"""
Created on Sun Feb 23 23:58:31 2020

@author: CK073783
"""

from model.models import Univariate
import json
import argparse
import pickle
import pandas as pd
from os import path,mkdir

save_path ='data/model_output/'

def filter_data(summary_df,data_object):
    return summary_df[(summary_df['valid'])&(summary_df['months_missing']==0)][data_object.target_items]



parser = argparse.ArgumentParser(description='Run Models for processed files')
parser.add_argument("-ti", "--target_item", 
                    required=True, 
                    help="Target_Item eg:- mention target item you want to run for or default give value as 'All' ")
parser.add_argument("-mn", "--model_name", required=True, 
                    default="All",
                    help="Specify model name to which you want to run by default its All")

with open('model-config.json') as f:
  model_config = json.load(f)

args = vars(parser.parse_args())



filehandler = open(model_config['processed_file'], 'rb') 
data_object = pickle.load(filehandler)

data_date = data_object.summary_df.end_date.max().month_name() +'-'+str(data_object.summary_df.end_date.max().year)
filtered_data  = data_object.transform_data[data_object.transform_data[data_object.target_items].isin(filter_data(data_object.summary_df,data_object))]
data_date = data_object.summary_df.end_date.max().month_name() +'-'+str(data_object.summary_df.end_date.max().year)



target_dates = data_object.target_dates
target_value = data_object.target_value
target_items = data_object.target_items

if args['target_item'] == 'All':
    tis = filtered_data[target_items].unique()

else:
    tis = [args['target_item']]

model_output = {}
for ti in tis:
    

    subset_df = filtered_data[filtered_data[target_items]==ti].reset_index(drop=True)
    subset_df[target_dates] = pd.to_datetime(subset_df[target_dates]).copy()
    
    
    subset_df = subset_df.sort_values('posted_date').reset_index()
    subset_df.set_index('posted_date',inplace=True)
    
    models = Univariate(ti,data = subset_df,target_items= target_items,target_value=target_value,target_dates=target_dates,model_name=args['model_name'])
    model_output[ti] = models
    
    
if path.exists(save_path+args['target_item'])==False:
    mkdir(save_path+args['target_item'])
    
obj_file = save_path+args['target_item']+'/'+data_date+'.pkl'
file =  open(obj_file,'wb')
pickle.dump(model_output,file)
file.close()