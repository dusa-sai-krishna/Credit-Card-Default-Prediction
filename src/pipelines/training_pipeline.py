import sys,os
from os.path import dirname,join,abspath
sys.path.insert(0,abspath(join(dirname(__file__),'../..')))

#import libraries
import pandas as pd
from src.logger import logging
from src.exception import CustomException

from src.components.data_ingestion import DataIngestion
from src.components.data_transformation import DataTransformation
#from src.components.model_trainer import ModelTrainer


if __name__=='__main__':
    ingestion_worker=DataIngestion()
    train_path,test_path=ingestion_worker.initiate_data_ingestion()
    print(train_path,test_path)
    
    transformer=DataTransformation()
    clean_train_arr,clean_test_arr=transformer.initiateDataTransformation(train_path,test_path)
    print('Data got cleaned successfully')
    
    #trainer=ModelTrainer()
    #trainer.initiateModelTrainer(clean_test_arr,clean_test_arr)
    #print('Model Trained Successfully')