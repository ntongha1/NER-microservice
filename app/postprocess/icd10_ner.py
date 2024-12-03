import json 
import torch 
from configs import current_config
from . import pipelines


def get_pipeline_class(pipeline_class_name):
    return getattr(pipelines, pipeline_class_name)

with open(current_config.config["current_config_path"]) as config_file:
    config = json.load(config_file)    

pipeline_class = get_pipeline_class(config["pipeline_class"])

pipeline = pipeline_class(config)

def get_icd10_codes_with_indices_from_text_slow(text):
 
    output_entities = pipeline(text)
            
    return  output_entities