#Dependencies
import boto3
import urllib
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, roc_curve, auc,classification_report
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report

from io import BytesIO
import uuid
import pickle
import mlflow
import xgboost as xgb


#useful functions
from physician_conversion_mlops.common import Task
from pyspark.sql import SparkSession
from pyspark.dbutils import DBUtils
from databricks import feature_store


class utils(Task):

    def push_df_to_s3(self,df,file_path):

        # AWS credentials and region
        aws_region = self.conf['s3']['aws_region']
        spark = SparkSession.builder.appName("CSV Loading Example").getOrCreate()

        dbutils = DBUtils(spark)

        aws_access_key = dbutils.secrets.get(scope="secrets-scope", key="aws-access-key")
        aws_secret_key = dbutils.secrets.get(scope="secrets-scope", key="aws-secret-key")
        
        
        # access_key = aws_access_key 
        # secret_key = aws_secret_key

        csv_buffer = BytesIO()
        df.to_csv(csv_buffer, index=False)
        csv_content = csv_buffer.getvalue()

        s3 = boto3.resource("s3",aws_access_key_id=aws_access_key, 
                    aws_secret_access_key=aws_secret_key, 
                    region_name=aws_region)

        #s3_object_key = self.conf['preprocessed']['preprocessed_df_path'] 
        s3.Object(self.conf['s3']['bucket_name'], file_path).put(Body=csv_content)

        return {"df_push_status": 'success'}
    

    def load_data_from_s3(self, bucket_name,file_path):

        # AWS credentials and region
        aws_region = self.conf['s3']['aws_region']
        

        spark = SparkSession.builder.appName("CSV Loading Example").getOrCreate()

        dbutils = DBUtils(spark)

        aws_access_key = dbutils.secrets.get(scope="secrets-scope", key="aws-access-key")
        aws_secret_key = dbutils.secrets.get(scope="secrets-scope", key="aws-secret-key")
        
        
        access_key = aws_access_key 
        secret_key = aws_secret_key

        print(f"Access key and secret key are {access_key} and {secret_key}")

        
        
        encoded_secret_key = urllib.parse.quote(secret_key,safe="")

        s3 = boto3.resource("s3",aws_access_key_id=aws_access_key, 
                      aws_secret_access_key=aws_secret_key, 
                      region_name=aws_region)
                

        s3_object = s3.Object(bucket_name, file_path)
        
        csv_content = s3_object.get()['Body'].read()

        df_input = pd.read_csv(BytesIO(csv_content))

        return df_input
    

    def select_kbest_features(self, df, target_col,n):
        """
        Selects the top n features from the DataFrame using the SelectKBest algorithm.

        Args:
            df: The DataFrame to select features from.
            n: The number of features to select.

        Returns:
            A list of the top n features.
        """


        selector = SelectKBest(k=n)
        selected_features = selector.fit_transform(df, target_col)
        
        mask = selector.get_support()
        top_n_features = df.columns[mask]

        return top_n_features
        
        
    def pickle_dump_list_to_s3(self, column_list):
        """
        Pickle dump a list of columns and upload it to an S3 bucket in the specified folder.

        Args:
        - column_list: List of columns to pickle.

        Returns:
        - upload pickle list to s3
        """
        # AWS details
        spark = SparkSession.builder.appName("CSV Loading Example").getOrCreate()

        bucket_name = self.conf['s3']['bucket_name']
        aws_region = self.conf['s3']['aws_region']
        folder_path = self.conf['preprocessed']['model_variable_list_file_path']
        file_name = self.conf['preprocessed']['model_variable_list_file_name']

        dbutils = DBUtils(spark)
        aws_access_key = dbutils.secrets.get(scope="secrets-scope", key="aws-access-key")
        aws_secret_key = dbutils.secrets.get(scope="secrets-scope", key="aws-secret-key")
        access_key = aws_access_key 
        secret_key = aws_secret_key
        print(f"Access key and secret key are {access_key} and {secret_key}")

        # Create an S3 client
        s3 = boto3.resource("s3",aws_access_key_id=aws_access_key, 
                      aws_secret_access_key=aws_secret_key, 
                      region_name=aws_region)

        # Pickle dump the list
        with open(file_name, 'wb') as file:
            pickle.dump(column_list, file)

        # Upload the pickled file to S3
        s3.Bucket(bucket_name).upload_file(file_name, folder_path + file_name)

        print(f"Pickled file '{file_name}' uploaded to S3 bucket '{bucket_name}' in folder '{folder_path}'.")

    
    def load_pickle_from_s3(self,bucket_name, file_path):
        try:
            # Create an S3 client
            aws_region = self.conf['s3']['aws_region']
            spark = SparkSession.builder.appName("CSV Loading Example").getOrCreate()
            dbutils = DBUtils(spark)
            aws_access_key = dbutils.secrets.get(scope="secrets-scope", key="aws-access-key")
            aws_secret_key = dbutils.secrets.get(scope="secrets-scope", key="aws-secret-key")
            access_key = aws_access_key 
            secret_key = aws_secret_key
            print(f"Access key and secret key are {access_key} and {secret_key}")

            s3 = boto3.resource("s3",aws_access_key_id=aws_access_key, 
                      aws_secret_access_key=aws_secret_key, 
                      region_name=aws_region)
                

            s3_object = s3.Object(bucket_name, file_path)

            # Read the pickle file from the S3 response
            pickle_data = s3_object.get()['Body'].read()

            # Deserialize the pickle data to obtain the Python object (list in this case)
            loaded_list = pickle.loads(pickle_data)

            return loaded_list
        except Exception as e:
            print(f"Error: {str(e)}")
            return None
        
    def convert_columns_to_string(self,df, columns):
        for col in columns:
            if col in df.columns:
                df[col] = df[col].astype(str)
            else:
                print(f"Column '{col}' not found in the DataFrame.")

    
    
    

    def eval_cm(self,model, X_train, y_train, X_val, y_val, drop_id_col_list):
        model.fit(X_train.drop(drop_id_col_list, axis=1, errors='ignore'), y_train)
        y_pred_train = model.predict(X_train.drop(drop_id_col_list, axis=1, errors='ignore'))
        y_pred_val = model.predict(X_val.drop(drop_id_col_list, axis=1, errors='ignore'))

        # Confusion Matrix
        plt.figure(figsize=(8, 6))
        cm_train = confusion_matrix(y_train, y_pred_train)
        cm_val = confusion_matrix(y_val, y_pred_val)
        plt.subplot(1, 2, 1)
        sns.heatmap(cm_train, annot=True, fmt='d', cmap='Blues', cbar=False)
        plt.title('Confusion Matrix (Train)')
        plt.savefig('confusion_matrix_train.png')
        plt.subplot(1, 2, 2)
        sns.heatmap(cm_val, annot=True, fmt='d', cmap='Blues', cbar=False)
        plt.title('Confusion Matrix (Validation)')
        plt.savefig('confusion_matrix_validation.png')
           
    
    
    def roc_curve(self,model, X_val,y_val, drop_id_col_list):
            
            """
            Logs Roc_auc curve in MLflow.

            Parameters:
            - y_test: The true labels (ground truth).
            

            Returns:
            - None
            """
            y_pred = model.predict(X_val.drop(drop_id_col_list, axis=1, errors='ignore'))
            fpr, tpr, thresholds = roc_curve(y_val, y_pred)
            roc_auc = roc_auc_score(y_val, y_pred)

            # Create and save the ROC curve plot
            plt.figure(figsize=(8, 6))
            plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
            plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title('Receiver Operating Characteristic (ROC) Curve')
            plt.legend(loc='lower right')
            roc_curve_plot_path = "roc_curve.png"
            
            plt.savefig(roc_curve_plot_path)


    def evaluation_metrics(self, model, X_train, y_train, X_val, y_val, drop_id_col_list):
        
        """
            Logs f1_Score and accuracy in MLflow.

            Parameters:
            - y_test: The true labels (ground truth).
            - y_pred: The predicted labels (model predictions).
            - run_name: The name for the MLflow run.

            Returns:
            - f1score and accuracy
        """

        model.fit(X_train.drop(drop_id_col_list, axis=1, errors='ignore'), y_train)
        y_pred_train = model.predict(X_train.drop(drop_id_col_list, axis=1, errors='ignore'))
        y_pred_val = model.predict(X_val.drop(drop_id_col_list, axis=1, errors='ignore'))

        f1_train = f1_score(y_train, y_pred_train)
        accuracy_train = accuracy_score(y_train, y_pred_train)

        f1_val = f1_score(y_val, y_pred_val)
        accuracy_val = accuracy_score(y_val, y_pred_val)

        return {'Train_F1-score' : round(f1_train,2),
                'Validation_F1-score' : round(f1_val,2),
                'Train_Accuracy' : round(accuracy_train,2),
                'Validation_Accuracy' : round(accuracy_val,2)}
        

    

            

