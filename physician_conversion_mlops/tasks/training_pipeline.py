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
        spark = SparkSession.builder.appName("FeatureStoreExample").getOrCreate()

        df_input = utils.load_data_from_s3(self,bucket_name, file_path)
        

        df_input = df_input.reset_index()
        df_input.drop(['index'], axis = 1, inplace = True, errors= 'ignore')

        #Clean column names
        df_input.columns = df_input.columns.str.strip()
        df_input.columns = df_input.columns.str.replace(' ', '_')

         #Convert ID columns to string type
        col_list = self.conf['feature_store']['lookup_key']
        utils.convert_columns_to_string(self,df_input, col_list)

        # Defining the features (X) and the target (y)
        X = df_input.drop("TARGET", axis=1)
        y = df_input["TARGET"]

        # Performing the train-test split to creat training df and inference set
        inference_size = self.conf['train_model_parameters']['inference_size']
        X_train_set, X_inference, y_train_set, y_inference = train_test_split(X, y,
                                                                       test_size=inference_size, 
                                                                       random_state=42,
                                                                         stratify= y)
        

        #Creating training df for model training by mering X_train, y_train
        X_train_df = pd.DataFrame(X_train_set)
        # y_train_df = pd.DataFrame(y_train_set)
        inference_df = pd.DataFrame(X_inference)

        #frames = [X_train_df,y_train_df]
        training_df = X_train_df.copy()
        training_df['TARGET'] = y_train_set
        training_df.drop(['index'], axis = 1, inplace = True, errors= 'ignore')

        #convert to spark dataframe with only Look-up key and Target for Featurelookup part
        training_df_spark = spark.createDataFrame(training_df)

        col_list_to_keep = self.conf['feature_store']['lookup_col_to_keep']

        feature_store_train_df = training_df_spark.select(*col_list_to_keep)

        #Save above datasets to s3
        file_path_infernece = self.conf['s3']['df_inference_set']
        utils.push_df_to_s3(self,inference_df,file_path_infernece)

        #load column list to be used for model training from from s3 
        bucket_name = self.conf['s3']['bucket_name']
        file_name = self.conf['s3']['model_variable_list_file_path']
        model_features_list = utils.load_pickle_from_s3(self,bucket_name, file_name)
        
        remove_list = ['TARGET','index']
        for i in remove_list:
            try:
                model_features_list.remove(i)
            except ValueError:
                pass
        print(len(model_features_list))
        

        model_feature_lookups = [
            FeatureLookup(
            table_name = self.conf['feature_store']['table_name'],
            feature_names = model_features_list,
            lookup_key = self.conf['feature_store']['lookup_key']
            )]
        
        
        
        # fs.create_training_set looks up features in model_feature_lookups that match the primary key from inference_data_df
        training_set = fs.create_training_set(
                        df=feature_store_train_df,
                        feature_lookups = model_feature_lookups,
                        label = self.conf['feature_store']['label'],
                        exclude_columns = self.conf['feature_store']['lookup_key']
                        )
        training_pd = training_set.load_df().toPandas()

        print('Training set created successfully')


        # Defining the features (X) and the target (y)
        X = training_pd.drop("TARGET", axis=1)
        y = training_pd["TARGET"]

        # Performing the train-test split to creat training df and inference set
        validation_size = self.conf['train_model_parameters']['val_size']
        X_train, X_val, y_train, y_val = train_test_split(X, y,
                                                                       test_size=validation_size, 
                                                                       random_state=42,
                                                                         stratify= y)
        

        #Creating training df for model training by mering X_train, y_train
        X_train_df = pd.DataFrame(X_train)
        X_val_df = pd.DataFrame(X_val)

        model_train_df =   X_train_df.copy()
        model_train_df['TARGET'] = y_train
        model_train_df.drop(['index'], axis = 1, inplace = True, errors= 'ignore')

        model_validation_df =  X_val_df.copy()
        model_validation_df['TARGET'] = y_val
        model_validation_df.drop(['index'], axis = 1, inplace = True, errors= 'ignore')

        #Save above datasets to s3
        file_path_training = self.conf['s3']['df_training_set']
        utils.push_df_to_s3(self,model_train_df,file_path_training)

        file_path_validation = self.conf['s3']['df_validation_set']
        utils.push_df_to_s3(self,model_validation_df,file_path_validation)  

        #train and log model using mlflow
        #client = MlflowClient()
        # run = client.create_run(experiment.experiment_id)
        # run = mlflow.start_run(run_id = run.info.run_id)
       
        mlflow.xgboost.autolog()
        mlflow.set_experiment(self.conf['mlflow']['experiment_name'])
        with mlflow.start_run() as run:
            
            params = self.conf['train_model_parameters']['model_params']
            drop_id_col_list = self.conf['feature_store']['lookup_key']

            model_xgb = xgb.XGBClassifier(**params, random_state=321)
            model_xgb.fit(X_train.drop(drop_id_col_list, axis=1, errors='ignore'), y_train)

            y_pred = model_xgb.predict(X_val.drop(drop_id_col_list, axis=1, errors='ignore'))

            #mlflow log models for reference
            mlflow.xgboost.log_model(
                xgb_model =model_xgb,
                artifact_path="usecase",
                # flavor=mlflow.xgboost,
                # training_set= training_set,
                registered_model_name="Physician_classifer",
                )

            #feature store log model for feature reference
            fs.log_model(
                model =model_xgb,
                artifact_path="usecase",
                flavor=mlflow.xgboost,
                training_set= training_set,
                registered_model_name="Physician_classifer"
            ) 

            #log confusion metrics
            utils.eval_cm(self,model_xgb, X_train, y_train, X_val,
                                            y_val,drop_id_col_list)
            
            # log roc curve
            utils.roc_curve(self,model_xgb, 
                            X_val,y_val,drop_id_col_list)
            
            #Log model evaluation metrics
            mlflow.log_metrics(utils.evaluation_metrics(
                self,model_xgb,
                X_train, y_train, 
                X_val, y_val,
                  drop_id_col_list))
            

            mlflow.log_artifact('confusion_matrix_train.png')
            mlflow.log_artifact('confusion_matrix_validation.png')
            mlflow.log_artifact('roc_curve.png')



    def launch(self):
            
            self.model_train()

    

def entrypoint():  
    
    task = Trainmodel()
    task.launch()

if __name__ == '__main__':
    entrypoint()
    