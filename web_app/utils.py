#header file
import sys,os
from os.path import dirname,join,abspath
sys.path.insert(0,abspath(join(dirname(__file__),'..')))

#from src.logger import logging
#from src.exception import CustomException

from sklearn.linear_model import LinearRegression,Lasso,Ridge,ElasticNet
from sklearn.metrics import r2_score,mean_absolute_error,mean_squared_error

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.base import BaseEstimator,TransformerMixin
from sklearn.preprocessing import MinMaxScaler




import pickle as pkl

def save_obj(file_path,obj):
    #logging.info(f'initiating to save the file at {file_path}')
    
    try:
        
        #get the directory name
        dirname=os.path.dirname(file_path)
        #logging.info(f'Obtained directory name to save the object,{dirname}')
        
        
        #create directory
        os.makedirs(dirname,exist_ok=True)
        #logging.info('successfully created the directory')        
        #save the model
        
        with open(file_path,'wb') as f:
            pkl.dump(obj,f)
        f.close()
        
        #logging.info('Successfully converted object to a pkl file')
        
    except Exception as e:
        #logging.info(f'Error while saving a obj, {e}')
        print(e)
        
        
def load_obj(file_path):
    #logging.info('Process of loading object started')
    
    try:
        with open(file_path,'rb') as f:
            obj=pkl.load(f)
        f.close()
        #logging.info(f'Object at {file_path} loaded successfully')
        return obj
        
    except Exception as e:
        #logging.info(f'Error occurred while loading object,{e}')
        print(e)






def model_evaluator(X_train,y_train,X_test,y_test,models):
    
    #logging.info('Model evaluation has started')
    
    try:
        model_report={}
        #iterate through models
        for name,model in models.items():
            
            #fit the model
            model.fit(X_train,y_train)
            
            #get predictions
            y_pred=model.predict(X_test)
            
            #get r2_score
            score=r2_score(y_test,y_pred)
            
            #update in model report
            model_report[model]=score

            #logging.info(f'Successfully evaluated {name} model')
        
        #logging.info('Model evaluation completed')
        return model_report
    except Exception as e:
        #logging.info(f'Error occurred during model evaluation {e}')
        print(e)
        
        
#---------------Custom Estimators--------------
class IdDropper(BaseEstimator,TransformerMixin):
    
    def fit(self,X,y=None):
        return self
    
    def transform(self,X,y=None):
        return X.drop('ID',axis=1)
    


class RemoveOutliers(BaseEstimator,TransformerMixin):
    
        
    def fit(self,X,y=None):
        return self
    
    def transform(self,X,y=None):
        
        def cap_upper(df,feature,limit):
    
            percentile_limit=df[feature].quantile(limit)
            df.loc[df[feature]>=percentile_limit,feature]=percentile_limit
            return df[feature]

        # Limit _Balance capping at 95%
        X['LIMIT_BAL']=cap_upper(X,'LIMIT_BAL',0.95)
        
        # Capping at 90%
        for col in ['BILL_AMT1', 'BILL_AMT2','BILL_AMT3', 'BILL_AMT4', 'BILL_AMT5', 'BILL_AMT6']:
            X[col]=cap_upper(X,col,0.90)
        
        for col in ['PAY_AMT1','PAY_AMT2', 'PAY_AMT3', 'PAY_AMT4', 'PAY_AMT5', 'PAY_AMT6']:
            X[col]=cap_upper(X,col,0.90)
        
        return X
    
    
class FeatureEncoder(BaseEstimator,TransformerMixin):
    
    def fit(self,X,y=None):
        return self
    
    def transform(self,X,y=None):
        
        # encoding sex
        if X['SEX'].dtype!='int64':
            X['SEX']=X['SEX'].map({'male':1,'female':2})
        
        # encoding education
        if X['EDUCATION'].dtype!='int64':
            X['EDUCATION']=X['EDUCATION'].map({'graduate school':1,'university':2,'high school':3,'others':4})
        # if education is already encoded then handling erroneous values
        X['EDUCATION']=X['EDUCATION'].apply(lambda x:str(x)).str.replace(r'[0456]','4')
        X['EDUCATION']=X['EDUCATION'].astype('int64')
        
        # encoding marriage
        if X['MARRIAGE'].dtype!='int64':
            X['MARRIAGE']=X['MARRIAGE'].map({'married':1,'single':2,'others':3})
        # if marriage is already encoded then handling erroneous values
        X['MARRIAGE']=X['MARRIAGE'].apply(lambda x:str(x)).str.replace(r'[30]',r'3')
        X['MARRIAGE']=X['MARRIAGE'].astype('int64')
        
        # repayment status
        for col in ['PAY_0', 'PAY_2', 'PAY_3', 'PAY_4', 'PAY_5', 'PAY_6']:
            X[col]=X[col].apply(lambda x: x if x>0 else 0) # if x is negative or zero then it's paid on time or in advance

        return X
    
class FeatureScaling(BaseEstimator,TransformerMixin):
    
    
    def fit(self,X,y=None):
        return self
    def transform(self,X,y=None):
        columns=['LIMIT_BAL', 'BILL_AMT1', 'BILL_AMT2', 'BILL_AMT3','BILL_AMT4', 'BILL_AMT5', 'BILL_AMT6', 'PAY_AMT1', 'PAY_AMT2','PAY_AMT3','PAY_AMT4', 'PAY_AMT5', 'PAY_AMT6' ]
        normalizer=MinMaxScaler()
        for col in columns:
            X[col]=normalizer.fit_transform(X[[col]])
        
        return X