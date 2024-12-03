env = '<ENV>'

microservice = "ICD Model"

# Health Status
class HealthStateConditions:
    HEALTHY = 'Healthy'
    DEGRADED = 'Degraded'
    UNHEALTHY = 'Unhealthy'
    ADVISORY = 'Advisory'

# s3 
aws_access_key_id = '<MODELS_AWS_ACCESS_KEY>'
aws_secret_access_key = '<MODELS_AWS_SECRET_KEY>'
region_name = '<MODELS_AWS_REGION>'
s3_model_bucket = '<MODELS_BUCKET_NAME>'
model_s3_folder = '<VR_SPARK_3_MODEL_FOLDER_NAME>'
icd10_s3_model_folder = 'icd-10-models'

local_models_folder = "pretrained_models"
local_data_folder = "./icd_model/data"

icd10_local_resources_folder = "models"

# HDFS folder
hdfs_sparkvr_directory = "/sparkvr"
hdfs_model_directory = f"{hdfs_sparkvr_directory}"
hdfs_data_directory = f"{hdfs_sparkvr_directory}/data"

#icd10_model = "enseble-ner-model.pt" # gpu: icd10-extraction-model.pt
s3_models_to_download = [
    "medicare_commercial_tag_encoder.pickle",
    "medicare_commercial_model_split.pt",
    "commercial_only_model_split.pt",
    "medicare_only_model_split.pt",
    "disease-negation-model.pt",
    "commercial_only_tag_encoder.pickle",
    "medicare_only_tag_encoder.pickle",
    "partial_overlap_logic.json",
    "hierarchy_rule.pickle"
]