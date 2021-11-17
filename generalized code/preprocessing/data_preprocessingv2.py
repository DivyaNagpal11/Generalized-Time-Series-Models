
"""
Created on Mon Dec  9 13:34:50 2019

@author: CK073783
"""


import pandas as pd
import numpy as np
import pickle
from os import path,mkdir,listdir
import holidays
import json

class DataPreprocessing:
    
    """
    data_path: specify path of the file
    not_include_be_list: list of Billing Entity that we should ommit
    overall_be: specify name of overall billing entity
    aggregate_choice: specify whether you need transformation Weekly or Monthly by default its daily
    dates: date column
    self.target_value: target column for which we need to predict
    items: granural columns by which we have to do prediction
    eg:- 
    target_dates = 'posted_date'
    target_value = 'total_charge'
    target_items = 'billing_entity'
    """
    
    # STEP1: Data Initialization 
    def __init__(self,data_path,not_include_be_list,overall_be,target_dates,target_value,target_items,target_file,other_target_value,aggregate_choice='Daily'):
        self.data_path = data_path
        self.target_file = target_file
        self.other_target_value = other_target_value
        self.data = pd.DataFrame()
        self.not_include_be_list = not_include_be_list
        self.overall_be = overall_be
        self.summary_df = pd.DataFrame()
        self.transform_data = pd.DataFrame()
        self.choice = aggregate_choice
        self.target_dates = target_dates
        self.target_value = target_value
        self.target_items = target_items
     

    
    #STEP 2: Data Conistency 
    def data_consistency_date(self,data):
        resample_data = data.set_index(pd.to_datetime(data[self.target_dates])).groupby([self.target_items]).resample('D').sum().reset_index()
        return resample_data
    
    #STEP 2.a: combine files
    def combine_data(self):
        DATA = {}
        files = [file.replace(".csv","") for file in listdir(self.data_path)]
        for file in files:            
            #Create data consistency
            DATA[file] = self.data_consistency_date(pd.read_csv(self.data_path+file+'.csv'))
            DATA[file][self.target_dates] = pd.to_datetime(DATA[file][self.target_dates])
            DATA[file].sort_values(self.target_dates).reset_index(drop=True,inplace=True)
        
        df = DATA[self.target_file]
        for file in DATA:
            if file != self.target_file:
                df = pd.merge(df,DATA[file],how='left',on=[self.target_dates,self.target_items])
        df.sort_values(self.target_dates).reset_index(drop=True,inplace=True)
        self.data = df.copy()
        return df
    
    #STEP 3: Create Overall Billing Entity
    def create_overall_be(self,data_update):
        new_df = data_update[~(data_update[self.target_items].isin(self.not_include_be_list))]
        all_advh=new_df.copy()
        all_advh[self.target_items]= self.overall_be 
        full_df=new_df.append(all_advh)
        return full_df
    
    #STEP 4: Create Granurlar columns
    def create_granular_col(self,df):
        df[self.target_dates]=pd.to_datetime(df[self.target_dates])
        df= df.sort_values(self.target_dates).reset_index(drop=True)
        df['Month']= df[self.target_dates].dt.month
        df['Year']= df[self.target_dates].dt.year
        df['Day']= df[self.target_dates].dt.day
        # df['Weekday']= df[self.target_dates].dt.weekday_name
        df['Weekday_num']= df[self.target_dates].dt.weekday
        df['MonthYear']= df['Month'].astype('str')+ '-' + df['Year'].astype('str')
        df['WeekNum']= df[self.target_dates].dt.week
        return df
    
    #Update summary file
    def update_summary(self,data,summary_df):
        #Sum of total charges per billing entity
        sum_data = data.groupby(self.target_items)[self.target_value].sum().reset_index()
               
        #yearly mean
        mean_data = data.set_index(pd.to_datetime(data[self.target_dates])).groupby(self.target_items).resample('Y')[self.target_value].sum().reset_index()
        mean_data = mean_data.groupby(self.target_items).mean().reset_index()
        mean_data.columns=['billing_entity','yearly_mean']
        sum_data.columns=['billing_entity','monthly_sum']
        
        summary_df = pd.merge(summary_df,sum_data,how='left',on=self.target_items)
        summary_df = pd.merge(summary_df,mean_data,how='left',on=self.target_items)
        
        describe_summary = pd.DataFrame(summary_df['monthly_sum'].describe())
        
        #Get Quantiles
        Q1 = describe_summary.loc['25%'].values[0]
        Q3 = describe_summary.loc['75%'].values[0]
        
        summary_df['BE_TYPE'] = ''
        #Assign types based on Quantiles
        summary_df.loc[(summary_df.monthly_sum>=Q3),'BE_TYPE'] = 'Large'
        summary_df.loc[(summary_df.monthly_sum<=Q1),'BE_TYPE'] = 'Small'
        summary_df.loc[(summary_df.monthly_sum<Q3) & (summary_df.monthly_sum>Q1),'BE_TYPE'] = 'Medium'
        
        #Check total counts of months
        full_data = data.groupby([self.target_items,'MonthYear'])
        full_data = full_data.MonthYear.first().groupby(level=0).size().reset_index()
        full_data.columns = [self.target_items,'Count of Months']
        full_data.sort_values(self.target_items)
        summary_df= pd.merge(summary_df,full_data,how='left',on=self.target_items)
        summary_df['valid'] = summary_df['Count of Months']>=9
        return summary_df
    
    #STEP 5: Check for partial data
    def remove_partial_data(self,modified_df,cut_off_day_start = 10 ,cut_off_day_end = 28 ):
        """
        Here we check for first and last month that has partial data, in case
        of partial data we will remove them and move it next month for first date
        
        if the Day is greater than cut_off_day_start then we are conisdering it is partial data
        if the Day is greater than cut_off_day_end then we are conisdering it is partial data
        """
       
        year = np.sort(list(set(modified_df.Year)))
        # Cut off Dates
        
        
        #Subset data with respect to year
        start_year_data = modified_df[modified_df.Year == year[0]].sort_values(self.target_dates).reset_index(drop=True) # start year data
        end_year_data = modified_df[modified_df.Year == year[-1]].sort_values(self.target_dates).reset_index(drop=True) # end year data
        
        start_date = start_year_data[self.target_dates].iloc[0]
        if (len(start_year_data.Month.unique())>1) & (start_date.month !=12):
            if start_date.day > cut_off_day_start:
                start_date = start_date+pd.offsets.MonthBegin()
        else:
            start_date = start_date+pd.offsets.MonthBegin()
            
        end_date = end_year_data[self.target_dates].iloc[-1]
        
        if end_date.month !=1:
            if end_date.day < cut_off_day_end:
                end_date = end_date-pd.offsets.MonthEnd()
        else:
            end_date = end_date-pd.offsets.MonthEnd()
    
           
        return start_date,end_date
    
    def subset_data(self,data):
                    
        summary_df = pd.DataFrame()
        new_full_df = pd.DataFrame()
        all_be_list = list(sorted(set(data[self.target_items])))
        for BE in all_be_list:
            modified_df = data[data[self.target_items]==BE]
            start_date,end_date = self.remove_partial_data(modified_df)    
            mask = (modified_df[self.target_dates] >= start_date) & (modified_df[self.target_dates] <= end_date)
            subset_df = modified_df.loc[mask]
            subset_df = subset_df.sort_values(by=self.target_dates).reset_index(drop=True)
            if len(subset_df)!=0:
                summary_df = summary_df.append(pd.DataFrame([BE,subset_df.iloc[0][self.target_dates],subset_df.iloc[-1][self.target_dates]]).T)
                new_full_df = new_full_df.append(subset_df)
        summary_df.columns=[self.target_items,'start_date','end_date']
        summary_df.reset_index(drop=True,inplace=True)
        current_date = summary_df.end_date.max()
        summary_df['months_missing'] = summary_df.end_date.apply(lambda row: (current_date.year-row.year)*12 + current_date.month-row.month)
        new_full_df = new_full_df.sort_values(by=self.target_dates).reset_index(drop=True)
        data = new_full_df
        
        
        return data,summary_df

    # GETS RAW DATA
    def  get_raw_data(self):
        if len(self.data)==0:
            return self.combine_data()
        return self.data
    
    # GETS Transformed data
    def get_transformed_data(self):
        """Gets transformed data returns summary_df & transformed data<br>
        
        transformed data: contains cleaned and transformed data
        summary_df: contains summary of data for each billing entity"""
        if len(self.transform_data)==0:
            self.transform()
        return self.summary_df,self.transform_data
    
    
    # Aggregate data monthly or weekly    
    def aggregate_data(self,data):
        if self.choice == "Monthly":
            trans_str='M'
        elif self.choice == "Weekly":
            trans_str='W'
        else:
            trans_str='D'
        
        return data[[self.target_dates, self.target_items, self.target_value]+self.other_target_value].set_index(pd.to_datetime(data[self.target_dates])).groupby([ self.target_items]).resample(trans_str).sum().reset_index()
    
    
    # Outlier detection & Negative charge count
    def count_outlier_negative(self,transformed_data,summary_df):
        g = transformed_data.groupby(by=self.target_items).apply(lambda row: row.quantile([.75,.25])).reset_index()
        a = g.pivot(index=self.target_items,columns='level_1',values=self.target_value)
        a.columns = ['Q1','Q3']
        a['IQR'] = a.Q3-a.Q1
        range_array = np.arange(1.5,2.5,.5)
        for i in range_array:     
            a['lower_bound_'+str(i)] = a.Q1-(a.IQR*i)
            a['upper_bound_'+str(i)] = a.Q3+(a.IQR*i)
        g = transformed_data.groupby(by=self.target_items)
        b =pd.merge(a,g[self.target_value].apply(lambda row: row.values),how='left',on=self.target_items).reset_index()
        b['negative_counts'] = b[self.target_value].apply(lambda row: len(row[row<0]))
        summary_df = pd.merge(summary_df,b[[self.target_items,'negative_counts']],how='left',on=self.target_items)
        for i in range_array:               
            b['count_outliers_'+str(i)] = b[[self.target_value,'upper_bound_'+str(i),'lower_bound_'+str(i)]].apply(lambda row: len(np.where(row[self.target_value]>row['upper_bound_'+str(i)])[0])+len(np.where(row[self.target_value]<row['lower_bound_'+str(i)])[0]),axis=1)
            summary_df = pd.merge(summary_df,b[[self.target_items,'count_outliers_'+str(i)]],how='left',on=self.target_items)
        
        return summary_df
    
    #create config file
    def create_config(self,obj_file):
        with open('data-config.json') as f:
            config_file = json.load(f)
        config_file['raw_data_path'] = self.data_path.replace("\\","/")
        config_file['processed_file'] = obj_file
        
        with open('model-config.json',"w") as f:
            f.write(json.dumps(config_file))
        f.close()
            
    
    #Save files
    def save_transformed_files(self):
        if len(self.transform_data)==0:
            self.transform()
        save_path ='data/processed_data/'
        data_date = self.summary_df.end_date.max().month_name()+'-'+str(self.summary_df.end_date.max().year)
        if path.exists(save_path)==False:
            mkdir(save_path+self.overall_be)
        obj_file = save_path+self.overall_be+'/'+data_date+'.pkl'
        file =  open(obj_file,'wb')
        pickle.dump(self,file)
        file.close()
        self.create_config(obj_file)
        
    def holidays(self,data):
        start_year = data[self.target_dates].min().year
        end_year = data[self.target_dates].max().year+2
        us_holidays=[]
        for date in holidays.UnitedStates(years=range(start_year,end_year)).items():
            us_holidays.append([str(date[0]),date[1]])
        
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
        
        merged = pd.DataFrame()
        
        for BE in data[self.target_items].unique():
            subset_data = data[data[self.target_items]==BE].copy().reset_index(drop=True)
            subset_data[['minor_holiday','major_holiday','observed_holiday']] = pd.merge(subset_data,us_holidays,on=self.target_dates,how='left')[['minor_holiday','major_holiday','observed_holiday']]
            merged = merged.append(subset_data)
        
        return merged
        
    # Transforms raw Data
    def transform(self):
        
        data_dc = self.combine_data()
        # print("\n","*"*20,"Data Consistency created","*"*20)
        
        #Create overall billing entity
        overall_be_data = self.create_overall_be(data_dc)
        del data_dc
        # print("\n","*"*20,"Overall BE created","*"*20)
        
        #Create granular columns
        granular_data = self.create_granular_col(overall_be_data)
        del overall_be_data
        # print("\n","*"*20,"Created Granural Columns","*"*20)
        
        #Remove partial data
        partial_removed_data,summary_df = self.subset_data(granular_data)
        del granular_data
        # print("\n","*"*20,"Removed partial months","*"*20)
        
        #Update Summary file with billing entity types and check for each of billing entity for minimum of 2 years
        summary_df = self.update_summary(partial_removed_data,summary_df)
        # print("\n","*"*20,"Updated summary data","*"*20)
        
        #Aggregate data 
        transformed_data = self.aggregate_data(partial_removed_data)
        # print("\n","*"*20,"Aggregated data","*"*20)
        
        #Anamoly detetction
        summary_df = self.count_outlier_negative(transformed_data,summary_df)
        # print("\n","*"*20,"Detected Anamolies & Outliers","*"*20)
        
        #Adding holiday columns
        holiday_data = self.holidays(transformed_data)
        
        #Save the transformation
        self.summary_df = summary_df
        self.transform_data = holiday_data
        
        del partial_removed_data
        del transformed_data
        del summary_df
        del holiday_data