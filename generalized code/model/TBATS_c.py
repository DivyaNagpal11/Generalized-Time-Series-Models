#Import files


from tbats import TBATS

import pandas as pd
import numpy as np

#Model class
class TBATS_c:

  

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



    def fit(self,data,param=None):
        estimator = TBATS(seasonal_periods=[3,12],
                     use_arma_errors=param[0],  # shall try models with and without ARMA
                     use_box_cox=param[1],  # will not use Box-Cox
                     use_trend=param[2],  # will try models with trend and without it
                     use_damped_trend=param[3],  # will try models with daming and without it
                     show_warnings=False,  # if set to False will not be showing any warnings for chosen model
                     )
        #######
        #   Fit funtion will fit your train data and return model
        #######
        return estimator.fit(data)




    def fitted_data(self,model,param=None):
        return model.y_hat 



    def predict(self,model,n_periods):
        return model.forecast(steps=n_periods) 



    def split_train_test(self,subset_df):
        return subset_df[:-3],subset_df[-3:]

    def mean_absolute_percentage_error(self,y_true,y_pred):
        return np.mean(np.abs(np.subtract(y_true,y_pred)/y_true))*100  

    def calculate_apes(self):
        for i,j in zip(self.test[self.target_value].values.flatten(),self.test_prediction.values.flatten()):            
            self.apes.append(self.mean_absolute_percentage_error(i,j))

    def optimize_tbats(self,parameters_list,train_set,val_set):

        val_set=val_set[self.target_value].to_numpy()
        results=[]
        best_adj_mape = float('inf')
        for param in parameters_list:
            try: 
                model=self.fit(train_set[self.target_value],param)
                fore1=self.predict(model,self.n_periods)
                fore=np.array(fore1)

                y_true=np.array(list(train_set[self.target_value]))
                y_pred=np.array(list(self.fitted_data(model)))
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
        test_predict = val_results.loc[:3]['test_prediction'].apply(lambda row: pd.Series(row).T).mean()

        self.test_prediction = pd.DataFrame(test_predict.values.flatten(),self.test.index,columns=[self.target_value])
        return val_results, best_model

      


    def run_model(self):
            
        val_results, best_model = self.optimize_tbats(self.parameters_list,self.train,self.test)
        #Model running for test prediction
        fitted_val_list=[]

        actual_data = self.train[self.target_value].append(self.test[self.target_value])
        for param in val_results['parameters'].loc[:5]:
            try:

                model = self.fit(actual_data,param)
                fore_test=self.predict(model,self.n_periods)
                fitted_val_list.append(fore_test.values.flatten())

            except:
                continue
        fitted_val=pd.DataFrame(fitted_val_list,columns=['1st','2nd','3rd'])
        fitted_mean=[fitted_val['1st'].mean(),fitted_val['2nd'].mean(),fitted_val['3rd'].mean()]

        unseen_index = pd.date_range(self.test.index[-1]+pd.tseries.offsets.MonthEnd(n=1),
              self.test.index[-1]+pd.tseries.offsets.MonthEnd(n=self.n_periods),freq='M')

        self.unseen_prediction = pd.DataFrame(fitted_mean,unseen_index,columns=[self.target_value])

        #Calculate apes for each test prediction        
        self.calculate_apes()
        self.train_model_params = val_results['parameters'].loc[:3]            
        self.test_model_params = val_results['parameters'].loc[:3]
        self.train_mape = val_results['train_mape'].loc[:3].mean()  
        self.test_mape = val_results['val_mape'].loc[:3].mean()

