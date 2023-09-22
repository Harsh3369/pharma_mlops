#system dependencies
import os
import datetime

#computation denendencies
import pandas as pd
import numpy as np

#useful functions
from physician_conversion_mlops.common import Task

#pyspark and feature store 
from pyspark.dbutils import DBUtils
from databricks.feature_store import feature_table, FeatureLookup

#warnings
warnings.filterwarnings('ignore')

class feature_pipeline(Task):
    def 