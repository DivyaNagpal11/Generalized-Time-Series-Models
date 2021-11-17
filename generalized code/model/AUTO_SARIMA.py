import pmdarima.arima as arima
import pandas as pd
import numpy as np

class AUTO_SARIMA():
             
    def __init__(self,**kwargs):
        
        self.random_state = kwargs.get('random_state',1)
        self.target_dates = kwargs.get('target_dates')
        self.target_value = kwargs.get('target_value')
        self.target_items = kwargs.get('target_items')
        self.train,self.test = kwargs.get('train')[self.target_value],kwargs.get('test')[self.target_value]
        self.start_p = kwargs.get('start_p',0)
        self.start_q = kwargs.get('start_q',0)
        self.start_P = kwargs.get('start_P',0)
        self.start_Q = kwargs.get('start_Q',0)
        
        self.max_p = kwargs.get('max_p',5)
        self.max_q = kwargs.get('max_q',5)
        self.max_P = kwargs.get('max_P',5)
        self.max_Q = kwargs.get('max_Q',5)
        self.n_periods = kwargs.get('n_periods',3)
        
        #output values
        self.fitted_values = pd.DataFrame()
        self.test_prediction = pd.DataFrame()
        self.unseen_prediction = pd.DataFrame()
        
        
        self.apes = []
        self.run_model()
        del self.train,self.test
        
    
    def fit(self,data):
        
        """
            Fit funtion will fit your train data and return model
        """
        
        return arima.auto_arima(self.train, error_action='ignore',seasonal=True,m=12,maxiter=1500,
                                       start_p=self.start_p,start_q=self.start_q,test='adf',
                                       max_p=self.max_p,max_q=self.max_q,start_P=self.start_P,
                                       start_Q=self.start_Q,max_P=self.max_P,max_Q=self.max_Q,
                                       suppress_warnings=True,stepwise=True,random_state=self.random_state)
    
    def fitted_data(self,model):
        return model.predict_in_sample()
    
    def predict(self,model):
        return model.predict(n_periods=self.n_periods)
    
    def mean_absolute_percentage_error(self,y_true,y_pred):
        return np.mean(np.abs(np.subtract(y_true,y_pred)/y_true))*100  

    def calculate_apes(self):
        for i,j in zip(self.test.values.flatten(),self.test_prediction.values.flatten()):            
            self.apes.append(self.mean_absolute_percentage_error(i,j))
            
    def run_model(self):
        #Model running for test prediction
        model = self.fit(self.train)
        self.fitted_values = pd.DataFrame(self.fitted_data(model),self.train.index,columns=[self.target_value])
        self.test_prediction = pd.DataFrame(self.predict(model),self.test.index,columns=[self.target_value])
        
        #Model running for unseen prediction
        model_test = self.fit(self.train.append(self.test))
        
        unseen_index = pd.date_range(self.test.index[-1]+pd.tseries.offsets.MonthEnd(n=1),
              self.test.index[-1]+pd.tseries.offsets.MonthEnd(n=self.n_periods),freq='M')
        
        self.unseen_prediction = pd.DataFrame(self.predict(model_test),unseen_index,columns=[self.target_value])
        
        #Calculate apes for each test prediction        
        self.calculate_apes()
        self.train_model_params = model.get_params()
        self.test_model_params = model_test.get_params()
        self.train_mape = self.mean_absolute_percentage_error(self.train.values.flatten(),self.fitted_values.values.flatten())
        self.test_mape = self.mean_absolute_percentage_error(self.test.values.flatten(),self.test_prediction.values.flatten())
