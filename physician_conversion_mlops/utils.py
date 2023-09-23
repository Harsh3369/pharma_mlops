#Dependencies
import boto3
import urllib
import pandas as pd
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif

from io import BytesIO
import uuid
import pickle


#useful functions
from physician_conversion_mlops.common import Task
from pyspark.sql import SparkSession
from pyspark.dbutils import DBUtils


class utils(Task):

    def push_df_to_s3(self,df):

        # AWS credentials and region
        aws_region = self.conf['s3']['aws_region']
        bucket_name = self.conf['s3']['bucket_name']
        file_path = self.conf['s3']['file_path']

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

        s3_object_key = self.conf['preprocessed']['preprocessed_df_path'] 
        s3.Object(self.conf['s3']['bucket_name'], s3_object_key).put(Body=csv_content)

        return {"df_push_status": 'success'}
    

    def load_data_from_s3(self):

        # AWS credentials and region
        aws_region = self.conf['s3']['aws_region']
        bucket_name = self.conf['s3']['bucket_name']
        file_path = self.conf['s3']['file_path']

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
        s3.upload_file(file_name, bucket_name, folder_path + file_name)

        print(f"Pickled file '{file_name}' uploaded to S3 bucket '{bucket_name}' in folder '{folder_path}'.")

        

