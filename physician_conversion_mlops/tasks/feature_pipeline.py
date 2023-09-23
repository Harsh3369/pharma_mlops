#system dependencies
import pandas as pd
import numpy as np


from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder
import warnings
import os
import boto3
import urllib
import pickle
from pyspark.sql import SparkSession
from io import BytesIO

#useful functions
from physician_conversion_mlops.common import Task

from physician_conversion_mlops.utils import utils

#pyspark and feature store 
import os
import datetime
from pyspark.dbutils import DBUtils

#warnings
warnings.filterwarnings('ignore')

# if __name__ == "__main__":
#     df_input = utils.load_data_from_s3()


class DataPrep(Task):    
  
    def _preprocess_data(self):
                
                df_input = utils.load_data_from_s3(self)

                df_input = df_input.reset_index()
        
                
                push_status = utils.push_df_to_s3(df_input)
                print(push_status)

                

    def launch(self):
         
         self._preprocess_data()

   

def entrypoint():  
    
    task = DataPrep()
    task.launch()


if __name__ == '__main__':
    entrypoint()

