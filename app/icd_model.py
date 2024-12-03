#!/usr/bin/env python
# coding: utf-8
import time
import sys
import json
import os, sys

import itertools
import pandas as pd, numpy as np

import constants as const
from utils.dapr_utils import update_encounter, update_encounter_status, update_medical_record_status, \
        get_icd_code_id, save_icd_codes, get_encounter_boundaries, get_encounter_ocr_pages, \
        get_workflow_ids_for_labels

from utils.file_utils import create_json_file, clean_local_folder, extract_gzip_file, load_json_file
from utils.hdfs_utils import upload_file_to_hdfs, upload_directory_to_hdfs, delete_directory_in_hdfs
from utils.s3_utils import download_file_from_s3_bucket_to_local_folder, upload_file
from utils.workflow_utils import (
    create_extracted_icd_codes_locations_object,
    get_icd_code_page_id_and_indices,
    save_extracted_icd_codes_location
)
import re


workflow_ids_dict = dict()
health_status = ''
replace_dict = ''

# icd_json_config = load_json_file('./configs', 'jan28_deployment_config.json')

# fix for icd10-extraction-model.pt bug: No module named 'src'
project_path = "/home/iquartic/"
sys.path.append(project_path)

def download_files_from_s3():
    # download files from s3
    print('download files from s3', flush=True)
    for model_file in const.s3_models_to_download:
        download_file_from_s3_bucket_to_local_folder(
            const.s3_model_bucket,
            const.icd10_s3_model_folder,
            model_file,
            const.icd10_local_resources_folder
        )

    # upload pipeline models and files to hdfs
    # upload_directory_to_hdfs(const.hdfs_model_directory, const.local_model_folder) 
    time.sleep(30)

# download files from s3
download_files_from_s3()

from postprocess import get_icd10_codes_with_indices_from_text_slow

def load_models_and_get_workflow_labels():
    global workflow_ids_dict, health_status
    from utils.dapr_utils import (
        update_microservice_state
    )
    from constants import HealthStateConditions
    try:

        # get workflow labels
        workflow_ids_dict = get_workflow_ids_for_labels()

        # update microservice health status to healthy
        update_microservice_state(HealthStateConditions.HEALTHY)
        health_status = "healthy"
    except Exception as e:
        print(e, flush=True)
        # update microservice health status to unhealthy
        update_microservice_state(HealthStateConditions.UNHEALTHY)


def find_icd_codes(id, text):
    print('find ICD codes', flush=True)
    """
    try:
        entities = get_icd10_codes_with_indices_from_text_slow(text)
        return entities
    except Exception as e:
        print(e, flush=True)
    """
    entities = get_icd10_codes_with_indices_from_text_slow(text)
    return entities


def extract_icd_codes_from_ocr_pages(
    encounter_id: str, 
    ocr_pages: dict, 
    boundaries:dict, 
    year: int, 
    plan: str, 
):
    """ Extract ICD Codes from OCR Pages"""
    global workflow_ids_dict, health_status
    if health_status is not "healthy":
        print("ICD Model health status is unhealthy", flush=True)
        return

    medical_record_id = ocr_pages[0].get("medicalRecordId")
    print(f"Extracting ICD codes for MR {medical_record_id} encounter {encounter_id}", flush=True)
    ocr_pages_text = ''
    ocr_page_ids = {}
    no_of_visit_pages = len(ocr_pages)
    print(f"no of ocr pages {no_of_visit_pages}", flush=True)
    visit_start_idx = boundaries["startPage"]["locations"][0]["startIdx"]
    visit_end_idx = boundaries["endPage"]["locations"][0]["endIdx"]

    for index, page in enumerate(ocr_pages):
        ocr_pages_text = ocr_pages_text + page.get("content")
        if no_of_visit_pages == 1:
            ocr_page_ids[str(0)] = page["Id"]
        else:
            ocr_page_ids[str(page["pageNumber"])] = page["Id"]

    visit_text = None
    if no_of_visit_pages == 1:
        visit_text = ocr_pages_text[visit_start_idx:visit_end_idx]
    elif no_of_visit_pages > 1:
        if ocr_pages_text.rfind("EasyOCRPageDividerStart") != -1 :
            last_start_page_divider_idx = ocr_pages_text.rfind("EasyOCRPageDividerStart")
            visit_text = ocr_pages_text[visit_start_idx:(last_start_page_divider_idx+visit_end_idx)]
    
    code_list = []

    if visit_text == None:
        print("visit text is None", flush=True)
    else:
        # code_list = find_icd_codes(encounter_id, remove_html_tags(ocr_pages_text))
        code_list = find_icd_codes(encounter_id, visit_text)

    print(f"No. of Extracted ICD codes for MR {medical_record_id} encounter {encounter_id} : {len(code_list)}", flush=True)

    if len(code_list) > 0:
        icd_codes_locations = dict()
        for extracted_code in code_list:
            try:
                icd_code = extracted_code['icd10_code']
                chunk = extracted_code['text']
                start_idx = extracted_code['start']
                end_idx = extracted_code['end']
                icd_code_page_id_and_indices = get_icd_code_page_id_and_indices(
                    visit_text=visit_text,
                    visit_start_index=visit_start_idx,
                    icd_code_start_idx=start_idx, 
                    icd_code_end_idx=end_idx, 
                    no_of_visit_pages=no_of_visit_pages, 
                    ocr_page_ids=ocr_page_ids
                )
                # print(icd_code_page_id_and_indices, flush=True)
                icd_codes_locations = create_extracted_icd_codes_locations_object(
                    icd_codes_locations,
                    icd_code=icd_code,
                    chunk=chunk,
                    ocr_page_id=icd_code_page_id_and_indices['page_id'],  
                    start_idx=icd_code_page_id_and_indices['start_idx'], 
                    end_idx=icd_code_page_id_and_indices['end_idx']
                )
                '''
                request_body = create_extracted_icd_codes_result_object(
                    icd_code=icd_code,
                    chunk=chunk,
                    ocr_page_id=icd_code_page_id_and_indices['page_id'],  
                    start_idx=icd_code_page_id_and_indices['start_idx'], 
                    end_idx=icd_code_page_id_and_indices['end_idx'], 
                    year=year
                )
                save_icd_codes(encounter_id, request_body)
                '''
            except Exception as e:
                print("Oops!", e.__class__, f" occurred. Unable to create location for {icd_code} for encounter {encounter_id}")
                print(e, flush=True)
        
        for icd_code in icd_codes_locations:
            try:
                save_extracted_icd_codes_location(encounter_id, icd_code, year, plan, icd_codes_locations[icd_code])
            except Exception as e:
                print("Oops!", e.__class__, f" occurred. Unable to save icd code {icd_code} for encounter {encounter_id}")
                print(e, flush=True)

    else:
        print(f"No ICD code found for MR {medical_record_id} encounter {encounter_id}", flush=True)

    clean_local_folder(const.local_data_folder)


def extract_icd_codes_for_encounters(
        medical_record_id: str, 
        encounter_ids: list, 
        year: int,
        plan: str,
        update_mr_status=True
    ):
    """ Extract ICD Codes for encounters"""
    global workflow_ids_dict, health_status
    if health_status is not "healthy":
        print("ICD Model health status is unhealthy", flush=True)
        return
    print(f"Performing ICD Extraction for encounters {encounter_ids}", flush=True)
    
    if update_mr_status:
        # update medical record status: icd extraction
        update_medical_record_status(workflow_ids_dict['icd_extraction_in_progress'], medical_record_id)

    for encounter_id in encounter_ids:
        try:
            ocr_pages = get_encounter_ocr_pages(encounter_id)
            print(f"Encounter id {encounter_id} ocr page size = {len(ocr_pages['pages'])}", flush=True)

            boundaries = get_encounter_boundaries(encounter_id)
            print(f"Encounter id {encounter_id} boundaries = {len(boundaries)}", flush=True)
            print(boundaries, flush=True)

            start_idx = boundaries["startPage"]["locations"][0]["startIdx"]
            end_idx = boundaries["endPage"]["locations"][0]["endIdx"]
            print(f"Encounter ID {encounter_id}  start idx {start_idx}, end idx {end_idx}", flush=True)

            extract_icd_codes_from_ocr_pages(encounter_id, ocr_pages['pages'], boundaries, year, plan)
            
            # update encounter status to set icd extracted flag to avoid multiple icd extraction: deprecated
            # ReviewTool uses 4 encounter statuses: New, In Progress, Complete, Deleted
            # update_encounter_status(workflow_ids_dict['icd_extraction_complete'], encounter_id)
            data = dict()
            data["icdExtracted"] = True
            update_encounter(encounter_id, data)
        except Exception as e:
            print(f"ICD Extraction failed for encounter {encounter_id}", flush=True)
            print(e, flush=True)
    
    if update_mr_status:
        update_medical_record_status(workflow_ids_dict['icd_extraction_complete'], medical_record_id)
        time.sleep(2)
        update_medical_record_status(workflow_ids_dict['final_review'], medical_record_id)

    return {'status': "icd extraction complete"}


def test_icd_extraction_using_encounters_and_mr_ids():
    """Test ICD Extraction using MR and encounter IDs"""
    print("Test icd extraction using Encounters and MRs IDs", flush=True)
    # download and load models plus labels
    load_models_and_get_workflow_labels()
    try:        
        year = 2020
        plan = "MA"
        mr_encounters = dict()
        encounters = ['61d44e3112eb87e2234359f7', '61d44e8012eb87e223435a0a']
        medical_record_id = '61d4478012eb87e2234359e6' 
        mr_encounters[medical_record_id] = encounters
        import time
        for medical_record_id in mr_encounters:
            encounters = mr_encounters[medical_record_id]
            try:
                extract_icd_codes_for_encounters(
                    medical_record_id=medical_record_id, 
                    encounter_ids=encounters, 
                    year=year,
                    plan=plan,
                    update_mr_status=True
                )
            except Exception as te:
                print("icd extraction error")
                print(te, flush=True)
            
            time.sleep(10)
    
    except Exception as e:
        print(e, flush=True)


def test_icd10_extraction_with_sample_text():
    """test ICD Extraction using sample text"""
    print("test ICD Extraction with sample text", flush=True)
    # download and load models plus labels
    load_models_and_get_workflow_labels()
    try:
        input_text = " ".join(['this is one sentense that CONTAIN ( ICD ) labels', 
        'lung CANCER (#%)',
        'heart beat problem colon cancer diabetes',
        'arthropathic psoriasis lantus humalog arthropatic psoriasis asthma prostate cancer cataracts type 2 diabetes mellitus \
                    peripheral neuropathy humalog type 2 diabetes mellitus hyperglycemia chronic kidney disease, stage 1 asthma malignant neoplasm \
                    prostate type 2 diabetes mellitus peripheral neuropathy type 2 diabetes mellitus hyperglycemia asthma malignant neoplasm prostate \
                    diabetic retinopathy mild nonproliferative'])

        entities = find_icd_codes("ecounter_id", input_text)
        print(entities, flush=True)
    except Exception as e:
        print("ICD extraction Exception")
        print(e, flush=True)