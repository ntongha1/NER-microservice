import glob
import json
from pathlib import Path
import os
import shutil
from zipfile import ZipFile
import tarfile


def unzip_pretrained_model(model_zipfile, model_folder):
    with ZipFile('pretrained_models/{}'.format(model_zipfile), 'r') as zipObj:
        zipObj.extractall('pretrained_models/{}/'.format(model_folder))
    # delete zip file 
    os.remove('pretrained_models/{}'.format(model_zipfile))

def delete_local_file(file):
    if os.path.isfile(file):
        os.remove(file)

def delete_csv_files():
    # delete csv files created by icd model
    dir = "./"
    for image_path in glob.iglob(os.path.join(dir, '*.csv')):
        os.remove(image_path)

def create_json_file(folder, filename, object:dict):
    if not os.path.exists(folder):
        os.makedirs(folder)
    file_path = os.path.join(folder, filename)
    
    # delete json file
    delete_local_file(file_path)
    
    # create json file
    Path(file_path).touch()
    with open(file_path, 'w') as input_file:
        input_file.write(json.dumps(object))

    return file_path

def load_json_file(dir, filename):
    print(f'load json file {filename}',flush=True)
    json_path = os.path.join(dir, filename)
    with open(json_path) as f:
        return json.load(f)

def clean_local_folder(folder):
    '''
    Delete all files in 'folder'
    '''
    try:
        if os.path.exists(folder) and os.path.isdir(folder):
            shutil.rmtree(folder)
            '''
            for filename in os.listdir(folder):
                file_path = os.path.join(folder, filename)
                if os.path.isfile(file_path):
                    os.unlink(file_path)
            '''
    except Exception:
        print(f"files not deleted in {folder}", flush=True)

def file_not_empty(filename): 
    # check if file is not empty
    return os.path.isfile(filename) and os.path.getsize(filename) > 0

def extract_gzip_file(file_name, folder):
    # extract tar file to proper location
    file = tarfile.open(file_name)
    file.extractall(folder)
    # file.extractall(os.path.join("/opt/spark/local",local_model_folder))
    file.close()