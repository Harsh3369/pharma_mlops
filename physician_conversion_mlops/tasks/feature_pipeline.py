#system dependencies
import pandas as pd
import numpy as np


from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder
from sklearn.feature_selection import VarianceThreshold
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif
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

                #Clean column names
                df_input.columns = df_input.columns.str.strip()
                df_input.columns = df_input.columns.str.replace(' ', '_')

                #Drop unwanted column: "HCO Affiliation" - "Affiliation Type" is more valid column for us
                drop_col_list = self.conf['feature_transformation']['drop_column_list']
                df_input.drop(drop_col_list, axis= 1, inplace= True)

                #One hot encode categorical features
                encode_col_list = self.conf['feature_transformation']['one_hot_encode_feature_list']
                df_input = pd.get_dummies(df_input, columns=encode_col_list, drop_first=True)

                #Select variables for feature selection
                id_target_col_list = self.conf['feature_transformation']['id_target_col_list']
                col_for_feature_selection = df_input.columns.difference(id_target_col_list)

                #Variance threshold feature selection method
                threshold = self.conf['param_values']['variance_threshold_value']
                var_thr = VarianceThreshold(threshold = threshold) #Removing both constant and quasi-constant
                var_thr.fit(df_input[col_for_feature_selection])
                #var_thr.get_support()

                df_input_subset = df_input[col_for_feature_selection]
                remove_col_list = [col for col in df_input_subset.columns 
                                if col not in df_input_subset.columns[var_thr.get_support()]]
                
                #remove above list column from master dataframe
                df_input.drop(remove_col_list, axis = 1, inplace = True, errors= 'ignore')

                #Feature Selection Using Select K Best
                n = self.conf['param_values']['select_k_best_feature_num']
                id_col_list = self.conf['feature_transformation']['id_col_list']
                target_col = self.conf['feature_transformation']['target_col']
                
                df = df_input.drop(id_col_list,axis=1)
                target_col_var = df_input[target_col]
                top_n_col_list = utils.select_kbest_features(self,
                      df,target_col_var, n)
                
                #Convert to list
                top_n_col_list = top_n_col_list.tolist()

                # Dump top_n_col_list to s3 bucket
                pickle_file_path = self.conf['preprocessed']['pickle_file_dump_list']
                utils.pickle_dump(self,top_n_col_list,pickle_file_path)
                
                #column list for dataframe
                cols_for_model_df_list = id_col_list + top_n_col_list
                df_feature_eng_output = df_input[cols_for_model_df_list]
                df_model_input = df_feature_eng_output.copy()
                
                push_status = utils.push_df_to_s3(self,df_model_input)
                print(push_status)

                

    def launch(self):
         
         self._preprocess_data()

   

def entrypoint():  
    
    task = DataPrep()
    task.launch()


if __name__ == '__main__':
    entrypoint()

