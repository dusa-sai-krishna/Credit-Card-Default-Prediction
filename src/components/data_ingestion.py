# Take data in and return splitted data
# import statements

import sys,os
from os.path import dirname,join,abspath
sys.path.insert(0,abspath(join(dirname(__file__),'../..')))

import pandas as pd
from sklearn.model_selection import train_test_split
from src.logger import logging
from src.exception import CustomException
from dataclasses import dataclass


#initialize the data ingestion configuration
@dataclass
class DataIngestionConfig():
    #defining class attributes
    data_path=os.path.join(os.getcwd(),"notebooks","data","UCI_Credit_Card.csv")
    train_path=os.path.join(os.getcwd(),"artifacts","train.csv")
    test_path=os.path.join(os.getcwd(),"artifacts","test.csv")
    raw_path=os.path.join(os.getcwd(),"artifacts","raw.csv")
    logging.info(f'Data path:{data_path}\n train_path:{train_path}\n test_path:{test_path}\n raw_path:{raw_path}')

#initialize a data ingestion class
class DataIngestion():
    
    def __init__(self):
        self.ingestion_config=DataIngestionConfig()
        
    def initiate_data_ingestion(self):
        logging.info('Initiating data ingestion')
        
        try:
           
           df=pd.read_csv(filepath_or_buffer=self.ingestion_config.data_path)
           logging.info(f'Data read successfully:,{df.head()}')
           
           #creating directory for saving raw data
           os.makedirs(os.path.dirname(self.ingestion_config.raw_path),exist_ok=True)#  very important
           logging.info('Directory to save the raw data is successfully created')
           
           #saving the file to  raw _path
           df.to_csv(self.ingestion_config.raw_path,index=False)
           logging.info('Raw file saved successfully')
           
           #splitting the data
           train_set,test_set=train_test_split(df,test_size=0.3,random_state=42)
           logging.info('Train test split successful')
           
           #save the data
           train_set.to_csv(self.ingestion_config.train_path,index=False,header=True)
           logging.info('Train set successfully saved')
           
           test_set.to_csv(self.ingestion_config.test_path,index=False,header=True)
           logging.info('Test set successfully saved')
           
           return (self.ingestion_config.train_path,self.ingestion_config.test_path)
       
        except Exception as e:
            logging.info(f'Data Ingestion error {e}')
            print(e)
        
        finally:
            print('Data Ingestion is performed')
            

#obj=DataIngestion()
#obj.initiate_data_ingestion()