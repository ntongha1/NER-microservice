from hdfs import InsecureClient
from constants import env

def dir_list(dir_path):
    '''directory list in hdfs '''
    try:
        client = InsecureClient(f'http://hdfs-httpfs.{env}.svc.cluster.local:14000', user='hdfs')
        for dir_content in client.list(dir_path):
            print(dir_content)
    except Exception as e:
        print("unable list directory in hdfs")
        print(e, flush=True)

def create_json_file_in_hdfs(file_name, json_data):
    '''create json file in hdfs'''
    try:
        client = InsecureClient(f'http://hdfs-httpfs.{env}.svc.cluster.local:14000', user='hdfs')
        with client.write(file_name, encoding='utf-8') as writer:
            from json import dump
            dump(json_data, writer)
    except Exception as e:
        print("unable to create json file in hdfs")
        print(e, flush=True) 

def upload_file_to_hdfs(hdfs_path, file_name):
    try:
        print("upload file to hdfs", flush=True)
        client = InsecureClient(f'http://hdfs-httpfs.{env}.svc.cluster.local:14000', user='hdfs')
        hdfs_folder_summary = client.content(hdfs_path, strict=False)
        if not hdfs_folder_summary:
            client.makedirs(hdfs_path)
        client.upload(hdfs_path, file_name)
    except Exception as e:
        print("failed to upload file to hdfs")
        print(e, flush=True)

def upload_directory_to_hdfs(hdfs_path, local_directory):
    try:
        print("upload directory to hdfs", flush=True)
        client = InsecureClient(f'http://hdfs-httpfs.{env}.svc.cluster.local:14000', user='hdfs')
        hdfs_folder_summary = client.content(hdfs_path, strict=False)
        if not hdfs_folder_summary:
            client.makedirs(hdfs_path)
        else:
            client.delete(hdfs_path, recursive=True, skip_trash=True)
        client.upload(hdfs_path, local_directory)
        print(hdfs_folder_summary, flush=True)
    except Exception as e:
        print("failed to upload directory to hdfs")
        print(e, flush=True)

def delete_directory_in_hdfs(hdfs_path):
    try:
        client = InsecureClient(f'http://hdfs-httpfs.{env}.svc.cluster.local:14000', user='hdfs')
        client.delete(hdfs_path, recursive=True, skip_trash=True)
    except Exception as e:
        print("failed to delete directory from hdfs")
        print(e, flush=True)
