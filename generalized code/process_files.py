#!/usr/bin/env python
# coding: utf-8

"""
Pass arguments that are needed to process files
argument 1: Dictionary file
argument 2: path of files that needs to be processed
"""

from preprocessing.data_preprocessingv2 import DataPreprocessing
import os
import sys
import json

with open('data-config.json') as f:
  config = json.load(f)

args = sys.argv

data_path = args[1].replace("\\","/")+"/"
not_include_be_list = config['not_include_be_list']
overall_be = config['overall_be']
aggregate_choice = config['aggregate_choice']
target_file = config['target_file']

target_dates = config['target_dates']
target_value = config['target_value']
target_items = config['target_items']
other_target_value = config['other_target_value']


dp = DataPreprocessing(data_path,not_include_be_list,overall_be,target_dates,
                        target_value,target_items,target_file,other_target_value,aggregate_choice) 
dp.save_transformed_files()

#process all data
# def process_data():
    # for file in os.listdir(raw_data_path):
        # data_path = raw_data_path+file+'/'
        # dp = DataPreprocessing(data_path,not_include_be_list,overall_be,target_dates,
                               # target_value,target_items,target_file,other_target_value,aggregate_choice) 
        # dp.save_transformed_files()
# # summary_df,transformed_data= dp.get_transformed_data()
# process_data()





