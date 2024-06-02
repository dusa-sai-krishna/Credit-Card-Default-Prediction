import os,sys
from os.path import dirname,join,abspath

sys.path.insert(0,abspath(join(dirname(__file__),'..')))


import logging,os
from datetime import datetime

#create a path for the directory to hold logs
LOG_PATH=os.path.join(os.getcwd(),"logs")

#create the directory
os.makedirs(LOG_PATH,exist_ok=True)

#specify the name of the log
LOG_FILE_NAME=f"{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.log"

#specify the log file path
LOG_FILE_PATH=os.path.join(LOG_PATH,LOG_FILE_NAME)

#config the logger
logging.basicConfig(filename=LOG_FILE_PATH,
                    level=logging.INFO,
                    format='%(asctime)s - %(name)s -  %(levelname)s - %(message)s)')
