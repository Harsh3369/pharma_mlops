#Basic ML dependencies
import pandas as pd
import numpy as np

#ML Training dependencies
from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder
from sklearn.feature_selection import VarianceThreshold
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif

#System and Env Dependencies
import warnings
import os
import boto3
import urllib
import pickle
from io import BytesIO
import datetime

#Spark and Databricks Dependencies
from pyspark.sql import SparkSession
from pyspark.dbutils import DBUtils
from databricks import feature_store
from databricks.feature_store import feature_table, FeatureLookup
from databricks.feature_store.online_store_spec import AmazonDynamoDBSpec

#useful functions
from physician_conversion_mlops.common import Task
from physician_conversion_mlops.utils import utils

#warnings
warnings.filterwarnings('ignore')


class DataPrep(Task):    
  
    def _preprocess_data(self):
                
                bucket_name = self.conf['s3']['bucket_name']
                file_path = self.conf['s3']['file_path']

                df_input = utils.load_data_from_s3(self,bucket_name,file_path)

                df_input = df_input.reset_index()

                #Clean column names
                df_input.columns = df_input.columns.str.strip()
                df_input.columns = df_input.columns.str.replace(' ', '_')
                
                #Convert ID columns to string type
                col_list = self.conf['feature_store']['lookup_key']
                utils.convert_columns_to_string(self,df_input, col_list)

                #Drop unwanted column: "HCO Affiliation" - "Affiliation Type" is more valid column for us
                drop_col_list = self.conf['feature_transformation']['drop_column_list']
                df_input.drop(drop_col_list, axis= 1, inplace= True)

                #One hot encode categorical features
                encode_col_list = self.conf['feature_transformation']['one_hot_encode_feature_list']
                df_input = pd.get_dummies(df_input, columns=encode_col_list, drop_first=True)

                #Clean column names
                df_input.columns = [c.replace('(', '').replace(')', '').replace(',', '')
                              .replace(';', '').replace('{', '').replace('}', '').replace('-', '')
                              .replace('\n', '').replace('\t', '').replace(' ', '_') 
                              for c in df_input.columns]
                
                print('columns name cleaned')
                print('')

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
                
                file_path = self.conf['preprocessed']['preprocessed_df_path']
                push_status = utils.push_df_to_s3(self,df_input,file_path) #saving the entire feature engineered data
                print(push_status)


                #Feature Selection Using Select K Best for training pipeline
                n = self.conf['param_values']['select_k_best_feature_num']
                id_col_list = self.conf['feature_transformation']['id_col_list']
                target_col = self.conf['feature_transformation']['target_col']
                
                df = df_input.drop(id_col_list,axis=1)
                target_col_var = df_input[target_col]
                top_n_col_list = utils.select_kbest_features(self,
                      df,target_col_var, n)
                
                #Convert to list
                top_n_col_list = top_n_col_list.tolist()

                # Dump top_n_col_list to s3 bucket to be used for training model
                utils.pickle_dump_list_to_s3(self,top_n_col_list)
                
                #column list for dataframe
                # cols_for_model_df_list = id_col_list + top_n_col_list
                # df_feature_eng_output = df_input[cols_for_model_df_list]
                # df_model_input = df_feature_eng_output.copy()
                
                
                #Save df_input to databricks feature store
                spark = SparkSession.builder.appName("FeatureStoreExample").getOrCreate()
                spark.sql(f"DROP TABLE IF EXISTS {self.conf['feature_store']['table_name']}")
                spark.sql(f"CREATE DATABASE IF NOT EXISTS {self.conf['feature_store']['table_name']}")


                df_feature = df_input.drop(target_col, axis = 1) #saving the entire features created
                df_spark = spark.createDataFrame(df_feature)

                fs = feature_store.FeatureStoreClient()

                # fs.drop_table(
                # name=self.conf['feature_store']['table_name']
                # )

                fs.create_table(
                        name=self.conf['feature_store']['table_name'],
                        df=df_spark,
                        primary_keys=self.conf['feature_store']['lookup_key'],
                        #labels=self.conf['feature_store']['label'],
                        schema=df_spark.schema,
                        description=self.conf['feature_store']['description']
                    )
                print("Feature Store is created")

                # # Overwrite mode does a full refresh of the feature table
                # fs.write_table(
                # name=self.conf['feature_store']['table_name'],
                # df = df_spark,
                # mode = 'overwrite'
                # )

                #publish the feature store
                online_store_spec = AmazonDynamoDBSpec(

                    region= 'us-west-2',

                    write_secret_prefix="feature-store-example-write/dynamo",

                    read_secret_prefix="feature-store-example-read/dynamo",

                    table_name = self.conf['feature_store']['table_name']

                    )
                fs.publish_table(self.conf['feature_store']['table_name'], online_store_spec)

                print("Feature Store published")

    def launch(self):
         
         self._preprocess_data()

   

def entrypoint():  
    
    task = DataPrep()
    task.launch()


if __name__ == '__main__':
    entrypoint()