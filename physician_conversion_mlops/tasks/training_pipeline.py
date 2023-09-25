#Basic ML dependencies
import pandas as pd
import numpy as np

#ML Training dependencies
from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder
from sklearn.feature_selection import VarianceThreshold
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score

# Importing necessary libraries for model development and evaluation
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, roc_curve, auc
import xgboost as xgb
import lightgbm as lgb
from urllib.parse import urlparse
import mlflow
from mlflow.tracking.client import MlflowClient

# Hyperparameter Tuning
from hyperopt import fmin, tpe, hp, SparkTrials, STATUS_OK, Trials

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


#useful functions
from physician_conversion_mlops.common import Task
from physician_conversion_mlops.utils import utils

#warnings
warnings.filterwarnings('ignore')


fs = feature_store.FeatureStoreClient()

class Trainmodel(Task):

    
    def model_train(self):
        bucket_name = self.conf['s3']['bucket_name']
        file_path = self.conf['s3']['file_path']

        df_input = utils.load_data_from_s3(self,bucket_name, file_path)

        df_input = df_input.reset_index()

        #Clean column names
        df_input.columns = df_input.columns.str.strip()
        df_input.columns = df_input.columns.str.replace(' ', '_')

        # Defining the features (X) and the target (y)
        X = df_input.drop("TARGET", axis=1)
        y = df_input["TARGET"]

        # Performing the train-test split to creat training df and inference set
        inference_size = self.conf['train_model_parameters']['inference_size']
        X_train, X_inference, y_train, y_inference = train_test_split(X, y,
                                                                       test_size=inference_size, 
                                                                       random_state=42,
                                                                         stratify= y)
        

        #Save above datasets to s3
        
        
        #Creating training df for model training by mering X_train, y_train
        X_df = pd.DataFrame(X)
        y_df = pd.DataFrame(y)

        frames = [X_df,y_df]
        training_df = pd.concat(frames)

        #load column list to be used for model training from from s3 
        bucket_name = self.conf['s3']['bucket_name']
        file_name = self.conf['s3']['model_variable_list_file_path']
        model_features_list = utils.load_pickle_from_s3(bucket_name, file_name)


        model_feature_lookups = [
            FeatureLookup(
            table_name = self.conf['feature_store']['table_name'],
            feature_names = model_features_list,
            lookup_key = self.conf['feature_store']['lookup_key']
            )]
        
        
        # fs.create_training_set looks up features in model_feature_lookups that match the primary key from inference_data_df
        training_set = fs.create_training_set(
                        df=training_df,
                        feature_lookups = model_feature_lookups,
                        label = self.conf['feature_store']['label'],
                        exclude_columns = self.conf['feature_store']['lookup_key']
                        )
        training_pd = training_set.load_df().toPandas()

        print('Training set created successfully')

    def launch(self):
            
            self.model_train()

    

def entrypoint():  
    
    task = Trainmodel()
    task.launch()

if __name__ == '__main__':
    entrypoint()
    