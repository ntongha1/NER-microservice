import os
import json
from typing import List, Optional
from fastapi import FastAPI, Request, Response
from pydantic import BaseModel
import threading

from icd_model import (
    load_models_and_get_workflow_labels, 
    extract_icd_codes_for_encounters,
    extract_icd_codes_from_ocr_pages, 
    test_icd_extraction_using_encounters_and_mr_ids, 
    test_icd10_extraction_with_sample_text
)

app = FastAPI()

class Encounter(BaseModel):
    encounter_id: str
    pages: List[dict]
    boundaries: dict
    year: int
    plan: str

class Encounters(BaseModel):
    encounterIds: List[str]
    medicalRecordId: str
    batchYear: int
    plan: str


def start_icd_extraction_from_ocr_pages(encounter: Encounter):
    """Start ICD Extraction using OCR Pages"""
    return extract_icd_codes_from_ocr_pages(
        encounter.encounter_id, 
        encounter.pages, 
        encounter.boundaries, 
        encounter.year, 
        encounter.plan,
    )


def start_icd_extraction_using_encounter_ids(encounters: Encounters):
    """Start ICD Extraction using Encounter IDs"""
    return extract_icd_codes_for_encounters(
        medical_record_id=encounters.medicalRecordId, 
        encounter_ids=encounters.encounterIds, 
        year=encounters.batchYear,
        plan=encounters.plan,
    )

threading.Timer(30, test_icd10_extraction_with_sample_text).start()
# threading.Timer(30, load_models_and_get_workflow_labels).start()

'''
@app.api_route("/ner-icd-extraction", methods=['OPTIONS'], status_code=200)
def icd_extraction():
    """
    Method to receive Dapr call to allow ner-icd-extraction input binding
    :return: json
    """
    print("Request to receive Dapr call to allow ner-icd-extraction input binding")
    return {'success': True}


@app.post(path="/ner-icd-extraction", status_code=200)
def ner_icd_extraction_using_ocr_pages(encounter: Encounter):
    # Method to extract encounter ICD labels
    # :return: json
    print(f"Request to extract ICD Codes for encounter {encounter.encounter_id}", flush=True)
    start_icd_extraction_from_ocr_pages(encounter)
    return {'success': True}
'''

@app.api_route("/ml-icd-extraction", methods=['OPTIONS'], status_code=200)
def ml_icd_extraction():
    """
    Method to receive Dapr call to allow ml-icd-extraction input binding
    :return: json
    """
    print("Request to receive Dapr call to allow ml-icd-extraction input binding")
    return {'success': True}


@app.post(path="/ml-icd-extraction", status_code=200)
def ml_icd_extraction_using_encounter_ids(encounters: Encounters):
    # Method to extract encounter ICD labels
    # :return: json
    print(f"Request to extract ICD Codes for encounters {encounters}", flush=True)
    return start_icd_extraction_using_encounter_ids(encounters)


@app.get("/prestop", status_code=200)
def prestop():
    """
    Hook for pod termination
    :return: json
    """
    print("ICD Model pod has terminated.", flush=True)
    # update microservice health status to degraded
    from utils.dapr_utils import update_microservice_state
    from constants import HealthStateConditions
    update_microservice_state(HealthStateConditions.DEGRADED)
    return {'success': True}


@app.get("/healthz", status_code=200)
def healthz():
    """
    This method returns http 200
    :return: json, object
    """
    return {'success': True}
