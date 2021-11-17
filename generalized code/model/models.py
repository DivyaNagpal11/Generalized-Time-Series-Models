# -*- coding: utf-8 -*-


import pandas as pd
import numpy as np
import time
from itertools import product
import warnings
warnings.filterwarnings('ignore')
pd.options.mode.chained_assignment = None

from .AUTO_ARIMA import AUTO_ARIMA
from .AUTO_SARIMA import AUTO_SARIMA
from .SARIMA import SARIMA
from .SARIMAX import SARIMAX
from .TBATS_c import TBATS_c

class Univariate():
    
    def split_train_test(self,subset_df):
        return subset_df[:-3],subset_df[-3:]
    
    
    def run_model(self,model_name,**kwargs):
        if model_name == 'auto.arima':
        # try:
            starttime = time.time()
            self.models_ran['auto.arima'] = AUTO_ARIMA(train = self.train,test = self.test,**kwargs)
            self.models_timetaken['auto.arima'] = time.time()-starttime
        # except:
        #     print('Dint run ARIMA model for ',self.target_items,' :',self.target_item)
        elif model_name == 'auto.sarima':
            try:
                starttime = time.time()
                self.models_ran['auto.sarima']  = AUTO_SARIMA(train = self.train,test = self.test,**kwargs)
                self.models_timetaken['auto.sarima'] = time.time()-starttime
            except:
                print('Dint run auto.sarima model for ',self.target_items,' :',self.target_item)
        elif model_name == 'sarima':
            try:
                starttime = time.time()
                p=range(0,4)
                d=range(0,2)
                q=range(0,4)
                P=range(0,3)
                D=range(0,2)
                Q=range(0,3)
                s=(3,6,12)
                
                parameters=product(p,d,q,P,D,Q,s)
                parameters_list=list(parameters)
                self.models_ran['sarima']  = SARIMA(train = self.train,test = self.test,parameters_list=parameters_list,**kwargs)
                self.models_timetaken['sarima'] = time.time()-starttime
            except:
                print('Dint run sarima model for ',self.target_items,' :',self.target_item)
        elif model_name == 'sarimax':
            try:
                starttime = time.time()
                p=range(0,4)
                d=range(0,2)
                q=range(0,4)
                P=range(0,3)
                D=range(0,2)
                Q=range(0,3)
                s=(3,6,12)
                
                parameters=product(p,d,q,P,D,Q,s)
                parameters_list=list(parameters)
                self.models_ran['sarimax']  = SARIMAX(train = self.train,test = self.test,parameters_list=parameters_list,**kwargs)
                self.models_timetaken['sarimax'] = time.time()-starttime
            except:
                print('Dint run sarimax model for ',self.target_items,' :',self.target_item)
        elif model_name == 'tbats':
            try:
                starttime = time.time()
                use_arma_errors=[True,False]
                use_box_cox=[True,False]
                use_trend=[True,False]
                use_damped_trend=[True,False]  
                parameters=product(use_arma_errors,use_box_cox,use_trend,use_damped_trend)
                parameters_list=list(parameters)
                self.models_ran['tbats']  = TBATS_c(train = self.train,test = self.test,parameters_list=parameters_list,**kwargs)
                self.models_timetaken['tbats'] = time.time()-starttime
            except:
                print('Dint run tbats model for ',self.target_items,' :',self.target_item)
            
    def __init__(self,target_item,**kwargs):
        self.random_state = kwargs.get('random_state',1)
        self.target_dates = kwargs.get('target_dates')
        self.target_value = kwargs.get('target_value')
        self.target_items = kwargs.get('target_items')
        self.model_name = kwargs.get('model_name','All')
        self.train,self.test = self.split_train_test(kwargs.get('data'))
        self.target_item = target_item
        self.models_ran = {}
        self.models_timetaken = {}

        if self.model_name == 'All':
            self.run_model('auto.sarima',**kwargs)
            self.run_model('auto.arima',**kwargs)
            self.run_model('sarima',**kwargs)               
            self.run_model('sarimax',**kwargs)   
            self.run_model('tbats',**kwargs)
        else:
            self.run_model(**kwargs)
                
               
        
        
             
           