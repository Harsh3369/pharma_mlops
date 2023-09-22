

def push_df_to_s3(self,df,access_key,secret_key):
            csv_buffer = BytesIO()
            df.to_csv(csv_buffer, index=False)
            csv_content = csv_buffer.getvalue()

            s3 = boto3.resource("s3",aws_access_key_id=access_key, 
                      aws_secret_access_key=secret_key, 
                      region_name='ap-south-1')

            s3_object_key = self.conf['preprocessed']['preprocessed_df_path'] 
            s3.Object(self.conf['s3']['bucket_name'], s3_object_key).put(Body=csv_content)

            return {"df_push_status": 'success'}