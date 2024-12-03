import os
import logging
import boto3
from botocore.exceptions import ClientError
from zipfile import ZipFile
from pathlib import Path
import constants as const


def download_file_from_s3_bucket(
        bucket,
        s3_folder,
        file,
        destfile
):
    # download file from s3 bucket
    # get a handle on s3
    print(f'download file {file} from s3 bucket ', flush=True)
    session = boto3.Session(aws_access_key_id=const.aws_access_key_id,
                            aws_secret_access_key=const.aws_secret_access_key,
                            region_name=const.region_name)
    s3 = session.resource('s3')
    try:
        s3.Bucket(bucket).download_file(f'{s3_folder}/{file}', destfile)
        print(f'{file} file downloaded from s3 bucket')
    except Exception as e:
        print("Oops!", e.__class__, "occurred. file not downloaded from s3 bucket", flush=True)
        print(e, flush=True)


def download_file_from_s3_bucket_to_local_folder(
        bucket: str,
        s3_folder: str,
        filename:  str,
        local_folder: str
):
    # download file from s3 bucket
    # get a handle on s3
    print(f'download file {filename} from s3 bucket {bucket}/{s3_folder} to local folder', flush=True)
    session = boto3.Session(aws_access_key_id=const.aws_access_key_id,
                            aws_secret_access_key=const.aws_secret_access_key,
                            region_name=const.region_name)
    s3 = session.resource('s3')
    try:
        Path(local_folder).mkdir(exist_ok=True)
        s3.Bucket(bucket).download_file(f'{s3_folder}/{filename}', os.path.join(local_folder, filename))
        print(f'{filename} file downloaded from s3 bucket')
    except Exception as e:
        print("Oops!", e.__class__, "occurred. file not downloaded from s3 bucket", flush=True)
        print(e, flush=True)


def download_and_extract_compressed_model_from_s3(s3_model_folder_name, zipfile_object_name, local_model_folder):
    # For S3, run this to download model zip files to your server and extract it to proper location
    session = boto3.Session(aws_access_key_id=const.aws_access_key_id,
                            aws_secret_access_key=const.aws_secret_access_key,
                            region_name=const.region_name)                  
    s3 = session.resource('s3')
    s3.Bucket(const.model_bucket_name).download_file('{}/{}'.format(s3_model_folder_name, zipfile_object_name), 'temp.zip')
    # extract file to proper location
    with ZipFile('temp.zip', 'r') as zipObj:
        zipObj.extractall('{}/'.format(local_model_folder))
    # delete zip file 
    os.remove('temp.zip')


def upload_file(aws_access_key_id, aws_secret_access_key, region_name, bucket, file_name, object_name=None):
    """Upload a file to an S3 bucket

    :param file_name: File to upload
    :param bucket: Bucket to upload to
    :param object_name: S3 object name. If not specified then file_name is used
    :return: True if file was uploaded, else False
    """

    # If S3 object_name was not specified, use file_name
    if object_name is None:
        object_name = file_name

    # Upload the file
    s3_client = boto3.client('s3',aws_access_key_id=aws_access_key_id,
                            aws_secret_access_key=aws_secret_access_key,
                            region_name=region_name)
    try:
        response = s3_client.upload_file(file_name, bucket, object_name)
    except ClientError as e:
        logging.error(e)
        return False
    return True