#%%
from tenacity import retry, stop_after_attempt, wait_fixed
from math import nan
from pathlib import Path
import ccxt
import json
import urllib.request
import config
import numpy as np
import pandas as pd
from dateutil import parser
from datetime import datetime
from time import sleep



import boto3

ec2 = boto3.client('ec2')
response = ec2.describe_instances()
print(len(response['Reservations'][0]['Instances']))
#%%
object_methods = [method_name for method_name in dir(ec2)
                  if callable(getattr(ec2, method_name))]
object_methods
#%%
import logging
import boto3
from botocore.exceptions import ClientError


def create_bucket(bucket_name, region=None):
    """Create an S3 bucket in a specified region

    If a region is not specified, the bucket is created in the S3 default
    region (us-east-1).

    :param bucket_name: Bucket to create
    :param region: String region to create bucket in, e.g., 'us-west-2'
    :return: True if bucket created, else False
    """

    # Create bucket
    try:
        if region is None:
            s3_client = boto3.client('s3')
            s3_client.create_bucket(Bucket=bucket_name)
        else:
            s3_client = boto3.client('s3', region_name=region)
            location = {'LocationConstraint': region}
            s3_client.create_bucket(Bucket=bucket_name,
                                    CreateBucketConfiguration=location)
    except ClientError as e:
        logging.error(e)
        return False
    return True

create_bucket('mybucket')
#%%
# Retrieve the list of existing buckets
s3 = boto3.client('s3')
response = s3.list_buckets()

# Output the bucket names
print('Existing buckets:')
for bucket in response['Buckets']:
    print(f'  {bucket["Name"]}')
# %%
