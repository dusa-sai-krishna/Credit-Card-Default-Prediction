#header file
import sys,os
from os.path import dirname,join,abspath
sys.path.insert(0,abspath(join(dirname(__file__),'../..')))

#import libraries
import os,sys
from src.logger import logging
from src.exception import CustomException
from src.utils import save_obj,IdDropper,RemoveOutliers,FeatureEncoder,FeatureScaling
from dataclasses import dataclass

import pandas as pd
import numpy as np

## pipelines
from sklearn.pipeline import Pipeline




#create a Data TransformationConfig Class
@dataclass
class DataTransformationConfig():
    preprocessor_file_path=os.path.join(os.getcwd(),"artifacts","preprocessor.pkl")
    clean_train_file_path=os.path.join(os.getcwd(),"artifacts","clean_train.csv")
    clean_test_file_path=os.path.join(os.getcwd(),"artifacts","clean_test.csv")
    

#create Data Transformation class
class DataTransformation():
    
    def __init__(self):
        self.transformation_config=DataTransformationConfig()
        
    
    def getPreprocessorObject(self):
        
        # Pipeline with custom estimators, it accepts a dataframe as input
        preprocessor=Pipeline(
        steps=[
            ('id_dropper',IdDropper()),
            ('rm_outlier',RemoveOutliers()),
            ('fe',FeatureEncoder()),
            ('fs',FeatureScaling())
        ]
    )
        return preprocessor
    
    def initiateDataTransformation(self,train_path,test_path):
        logging.info('Data Transformation has started')
        try:
            
            
            #read test and train data
            train_df=pd.read_csv(train_path)
            logging.info('Train data read successfully')
            
            test_df=pd.read_csv(test_path)
            logging.info('Test data read successfully')
            
            #split dependent and independent features
            X_train,y_train=train_df.drop(['default.payment.next.month'],axis=1),train_df['default.payment.next.month']
            X_test,y_test=test_df.drop(['default.payment.next.month'],axis=1),test_df['default.payment.next.month']
            logging.info('Splitting of Dependent and Independent features is successful')
            
            # get preprocessor and pre-process the content
            preprocessor=self.getPreprocessorObject()
            X_train_clean=preprocessor.fit_transform(X_train) # returns a dataframe
            logging.info('X_train successfully pre-processed')
            
            X_test_clean=preprocessor.transform(X_test)# returns a dataframe
            logging.info('X_test successfully pre-processed')
            
            #combine X_train_arr with y_train and vice versa
            clean_train_df=pd.concat([X_train_clean,y_train],axis=1)
            clean_test_df=pd.concat([X_test_clean,y_test],axis=1)
            logging.info('Concatenation of  cleaned arrays is successful')
            
            #save the pre-processor 
            save_obj(self.transformation_config.preprocessor_file_path,preprocessor)
            logging.info('Pre-processor successfully saved')
            
            return(
                clean_train_df,clean_test_df
            )
            
            
                    
        except CustomException as e:
            logging.info(f'Exception occurred in Data Transformation,{e}')
            print(e)
