#Import files


import statsmodels.api as sm
import holidays

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')
pd.options.mode.chained_assignment = None


#Model class
class SARIMAX:

  

    def __init__(self,**kwargs):

        #input parameters

        self.train,self.test = kwargs.get('train'),kwargs.get('test')       
        self.random_state = kwargs.get('random_state',1)
        self.target_dates = kwargs.get('target_dates')
        self.target_value = kwargs.get('target_value')
        self.target_items = kwargs.get('target_items')
        
        self.parameters_list = kwargs.get('parameters_list',None)

        self.n_periods = kwargs.get('n_periods',3)

        #output values
        self.fitted_values = pd.DataFrame()
        self.test_prediction = pd.DataFrame()
        self.unseen_prediction = pd.DataFrame()        
        self.apes = []
        self.run_model()
        del self.train,self.test



    def fit(self,data,exog_data,param=None):
            
        ###########
        #  Fit funtion will fit your train data and return model
        ###########
        return sm.tsa.statespace.SARIMAX(data,exog=np.asarray(exog_data),order=(param[0],param[1],param[2]),
                                                seasonal_order=(param[3],param[4],param[5],param[6])).fit(disp=-1)




    def fitted_data(self,model,param=None):
        return model.fittedvalues[param[6]+param[1]:] 



    def predict(self,model,exog_data,start,n_periods):
        return model.predict(exog=exog_data,start=start,end=start+n_periods-1) 



    def split_train_test(self,subset_df):
        return subset_df[:-3],subset_df[-3:]

    def mean_absolute_percentage_error(self,y_true,y_pred):
        return np.mean(np.abs(np.subtract(y_true,y_pred)/y_true))*100  

    def calculate_apes(self):
        for i,j in zip(self.test[self.target_value].values.flatten(),self.test_prediction.values.flatten()):            
            self.apes.append(self.mean_absolute_percentage_error(i,j))

    def forecast_exogs(self,val_set):

        max_date = val_set.index.max()
        unseen_dates = pd.DataFrame()
        unseen_dates[self.target_dates] = pd.date_range(max_date+pd.tseries.offsets.MonthEnd(n=1),
                  max_date+pd.tseries.offsets.MonthEnd(n=self.n_periods),freq='M')

        start_year = val_set.index.max().year
        end_year = start_year+2
        us_holidays=[]
        for date in holidays.UnitedStates(years=range(start_year,end_year)).items():
            us_holidays.append([str(date[0]),date[1]])

        us_holidays=pd.DataFrame(us_holidays,columns=[self.target_dates,'holiday'])
        us_holidays=pd.DataFrame(us_holidays,columns=[self.target_dates,'holiday'])
        us_holidays[self.target_dates]=pd.to_datetime(us_holidays[self.target_dates])
        us_holidays.holiday=us_holidays.holiday.astype(str)
        us_holidays['minor_holiday']=0
        minor_holidays = ['Martin Luther King, Jr. Day' ,"Washington's Birthday" ,'Columbus Day' ,'Veterans Day'] 
        us_holidays.loc[us_holidays.holiday.isin(minor_holidays),'minor_holiday']=1
        us_holidays['major_holiday']=0
        major_holidays = ["New Year's Day" ,"Christmas Day",'Thanksgiving','Memorial Day','Labor Day','Independence Day'] 
        us_holidays.loc[us_holidays.holiday.isin(major_holidays),'major_holiday']=1
        us_holidays['observed_holiday']=0
        observed =  us_holidays.holiday.apply(lambda row: True if row[1].endswith("(Observed)") else False)
        us_holidays.loc[observed,'observed_holiday']=1
        us_holidays = us_holidays.convert_dtypes()
        us_holidays.sort_values(self.target_dates).reset_index(drop=True,inplace=True)
        us_holidays = us_holidays.set_index(self.target_dates).resample('M').sum().reset_index()

        return pd.merge(unseen_dates,us_holidays,on=self.target_dates,how='left')[['minor_holiday','major_holiday','observed_holiday']]

    def optimize_SARIMAX(self,parameters_list,train_set,val_set,exog_columns):

        val_set1 = val_set.copy()
        val_set=val_set[self.target_value].to_numpy()
        results=[]
        best_adj_mape = float('inf')
    #         self.train = self.train.astype(float)
        for param in parameters_list:
            try:
                model=self.fit(train_set[self.target_value].astype(float),train_set.loc[:,exog_columns].astype(float).reset_index(drop=True),param)
                fore1=self.predict(model,val_set1.loc[:,exog_columns].astype(float).reset_index(drop=True),self.train.shape[0],self.n_periods)
                fore=np.array(fore1)

                y_true=np.array(list(train_set[param[6]+param[1]:][self.target_value]))
                y_pred=np.array(list(model.fittedvalues[param[6]+param[1]:]))
                train_mape=round((self.mean_absolute_percentage_error(y_true,y_pred)),2)
                val_mape=round((self.mean_absolute_percentage_error(val_set,fore)),2)
                adj_mape = train_mape*len(y_true)/(len(y_true)+len(val_set))+val_mape*len(val_set)/(len(y_true)+len(val_set))
                if adj_mape <= best_adj_mape:
                    best_adj_mape=adj_mape
                    best_model = model    
                results.append([param,model.aic,train_mape,val_mape,adj_mape,fore])
            except:
                continue

        result_table=pd.DataFrame(results)

        result_table.columns=['parameters','aic','train_mape','val_mape','adj_mape','test_prediction']
        val_results = result_table.sort_values(by='adj_mape',ascending=True).reset_index(drop=True).copy()
        test_predict = val_results.loc[:5]['test_prediction'].apply(lambda row: pd.Series(row).T).mean()

        self.test_prediction = pd.DataFrame(test_predict.values.flatten(),self.test.index,columns=[self.target_value])
        return val_results, best_model

      


    def run_model(self):
        exog_columns =  ['minor_holiday','major_holiday', 'observed_holiday']
        val_results, best_model = self.optimize_SARIMAX(self.parameters_list,self.train,self.test,exog_columns)
        #Model running for test prediction
        fitted_val_list=[]

        actual_data = self.train.append(self.test)
        for param in val_results['parameters'].loc[:5]:
            try:

                model = self.fit(actual_data[self.target_value],actual_data.loc[:,exog_columns].astype(float).reset_index(drop=True),param)
                fore_test=self.predict(model,self.forecast_exogs(self.test).astype(float).reset_index(drop=True),actual_data.shape[0],self.n_periods)
                fitted_val_list.append(fore_test.values.flatten())

            except:
                continue

        fitted_val=pd.DataFrame(fitted_val_list,columns=np.arange(1,self.n_periods+1))
        fitted_mean=[]
        for i in np.arange(1,self.n_periods+1):
            fitted_mean.append(fitted_val[i].mean())

        unseen_index = pd.date_range(self.test.index[-1]+pd.tseries.offsets.MonthEnd(n=1),
              self.test.index[-1]+pd.tseries.offsets.MonthEnd(n=self.n_periods),freq='M')

        self.unseen_prediction = pd.DataFrame(fitted_mean,unseen_index,columns=[self.target_value])

        #Calculate apes for each test prediction        
        self.calculate_apes()
        self.train_model_params = val_results['parameters'].loc[:5]            
        self.test_model_params = val_results['parameters'].loc[:5]
        self.train_mape = val_results['train_mape'].loc[:5].mean()  
        self.test_mape = val_results['val_mape'].loc[:5].mean()


