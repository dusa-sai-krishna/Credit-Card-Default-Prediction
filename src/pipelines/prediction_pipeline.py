#header imports
import sys,os
from os.path import dirname,join,abspath

from src.utils import load_obj,IdDropper,RemoveOutliers,FeatureEncoder,FeatureScaling
sys.path.insert(0,abspath(join(dirname(__file__),'../..')))

from src.logger import logging
from src.exception import CustomException
from dataclasses import dataclass

import pandas as pd

#create Prediction Pipeline config class
@dataclass
class PredictionPipelineConfig():
    
    preprocessor_file_path=os.path.join(os.getcwd(),"artifacts","preprocessor.pkl")
    model_file_path=os.path.join(os.getcwd(),"artifacts","model.pkl")
    

#initiate PredictionPipeline
class PredictionPipeline():
    
    def __init__(self) -> None:
        self.predictionPipeline_config=PredictionPipelineConfig()
    
    def predict(self,features):
        
        # load the pre-processor
        preprocessor=load_obj(self.predictionPipeline_config.preprocessor_file_path)
        logging.info('Preprocessor loaded successfully')

        
        #load the model
        model=load_obj(self.predictionPipeline_config.model_file_path)
        logging.info('Model loaded successfully')
        
        #preprocess the features
        cleaned_features=preprocessor.transform(features)
        logging.info(f'Features are cleaned:{cleaned_features}')

        # selecting imp features
        cleaned_features=cleaned_features[['PAY_0', 'PAY_2', 'PAY_4', 'PAY_3', 'PAY_6']]
        
        #get prediction
        prediction=model.predict(cleaned_features)
        logging.info(f'Prediction is successful {prediction}')
        
        return prediction
    
    

class CustomData:
    def __init__(self,
                 ID,
                 LIMIT_BAL,
                 SEX,
                 EDUCATION,
                 MARRIAGE,
                 AGE,
                 PAY_0,
                 PAY_2,
                 PAY_3,
                 PAY_4,
                 PAY_5,
                 PAY_6,
                 BILL_AMT1,
                 BILL_AMT2,
                 BILL_AMT3,
                 BILL_AMT4,
                 BILL_AMT5,
                 BILL_AMT6,
                 PAY_AMT1,
                 PAY_AMT2,
                 PAY_AMT3,
                 PAY_AMT4,
                 PAY_AMT5,
                 PAY_AMT6,
                 ):

        self.ID = ID
        self.LIMIT_BAL = LIMIT_BAL
        self.SEX = SEX
        self.EDUCATION = EDUCATION
        self.MARRIAGE = MARRIAGE
        self.AGE = AGE
        self.PAY_0 = PAY_0
        self.PAY_2 = PAY_2
        self.PAY_3 = PAY_3
        self.PAY_4 = PAY_4
        self.PAY_5 = PAY_5
        self.PAY_6 = PAY_6
        self.BILL_AMT1 = BILL_AMT1
        self.BILL_AMT2 = BILL_AMT2
        self.BILL_AMT3 = BILL_AMT3
        self.BILL_AMT4 = BILL_AMT4
        self.BILL_AMT5 = BILL_AMT5
        self.BILL_AMT6 = BILL_AMT6
        self.PAY_AMT1 = PAY_AMT1
        self.PAY_AMT2 = PAY_AMT2
        self.PAY_AMT3 = PAY_AMT3
        self.PAY_AMT4 = PAY_AMT4
        self.PAY_AMT5 = PAY_AMT5
        self.PAY_AMT6 = PAY_AMT6
        

    def get_data_as_dataframe(self):
        try:
            custom_data_input_dict = {
                'ID': [self.ID],
                'LIMIT_BAL': [self.LIMIT_BAL],
                'SEX': [self.SEX],
                'EDUCATION': [self.EDUCATION],
                'MARRIAGE': [self.MARRIAGE],
                'AGE': [self.AGE],
                'PAY_0': [self.PAY_0],
                'PAY_2': [self.PAY_2],
                'PAY_3': [self.PAY_3],
                'PAY_4': [self.PAY_4],
                'PAY_5': [self.PAY_5],
                'PAY_6': [self.PAY_6],
                'BILL_AMT1': [self.BILL_AMT1],
                'BILL_AMT2': [self.BILL_AMT2],
                'BILL_AMT3': [self.BILL_AMT3],
                'BILL_AMT4': [self.BILL_AMT4],
                'BILL_AMT5': [self.BILL_AMT5],
                'BILL_AMT6': [self.BILL_AMT6],
                'PAY_AMT1': [self.PAY_AMT1],
                'PAY_AMT2': [self.PAY_AMT2],
                'PAY_AMT3': [self.PAY_AMT3],
                'PAY_AMT4': [self.PAY_AMT4],
                'PAY_AMT5': [self.PAY_AMT5],
                'PAY_AMT6': [self.PAY_AMT6],
                
            }
            df = pd.DataFrame(custom_data_input_dict)
            logging.info('DataFrame Gathered')
            return df
        except Exception as e:
            logging.info('Exception Occurred in prediction pipeline')
            raise CustomException(e, sys)