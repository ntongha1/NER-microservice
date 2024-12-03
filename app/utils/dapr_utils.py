import ast 
import datetime
import json
from dapr.clients import DaprClient


def invoke_microservice_using_service_invocation(microservice: str, endpoint: str, http_verb:str, data: dict=None):
    """
    This method invokes microservice using service invocation
    :return: json
    """ 
    print(f"Invoke {microservice} using service invocation", flush=True)
    with DaprClient() as dapr_client:
        resp = dapr_client.invoke_method(
            app_id=microservice,
            method_name=endpoint,
            http_verb=http_verb,
            http_querystring='',
            data=json.dumps(data, indent=4) if data else '',
            content_type='application/json', 
        )
        return json.loads(resp.text())

def dapr_get_service_invocation(endpoint: str):
    """
    This method invokes Workflow Microservice to get data
    """ 
    print("Invoke Workflow Microservice to get data : \"{}\"".format(endpoint), flush=True)
    return invoke_microservice_using_service_invocation('workflow-microservice', endpoint, 'GET')

def save_data_using_service_invocation(endpoint: str, data: dict):
    """
    This method invokes Workflow Microservice to save data to MongoDB
    :return: InvokeServiceResponse
    """ 
    print("Invoke Workflow Microservice to save data in MongoDB",flush=True)
    return invoke_microservice_using_service_invocation('workflow-microservice', endpoint, 'POST', data)

def update_data_using_service_invocation(endpoint: str, data: dict):
    """
    This method invokes Workflow Microservice to update data in MongoDB
    :return: InvokeServiceResponse
    """ 
    print("Invoke Workflow Microservice to update data in MongoDB",flush=True)
    return invoke_microservice_using_service_invocation('workflow-microservice', endpoint, 'PUT', data)


def get_ocr_pages_by_encounter_id_from_workflow_microservice(encounter_id: str):
    """
    This method gets OCR Pages by encounter ID
    """ 
    print("Invoke Workflow Microservice to get OCR Pages by encounter ID")
    json_response = invoke_microservice_using_service_invocation('workflow-microservice', f'dapr-get-ocr-pages-by-encounter-id/{encounter_id}', 'GET')
    # labels = ast.literal_eval(resp.text()).get("labels")
    return json_response["pages"]

def save_icd_labels_by_encounter_id_using_service_invocation(encounter_id: str, icd_labels: dict):
    """
    This method invokes Workflow Microservice to save encounter ICD labels
    :return: InvokeServiceResponse
    """ 
    print("Invoke Workflow Microservice to save encounter ICD labels",flush=True)
    response = update_data_using_service_invocation("dapr-save-icd-labels/{}".format(encounter_id), icd_labels)
    return response

def update_encounter(encounter_id: str, data: dict):
    """
    This method invokes Workflow Microservice to update Encounter
    :return: json
    """ 
    print("Invoke Workflow Microservice to update Encounter{} with {}".format(encounter_id, data), flush=True)
    response = invoke_microservice_using_service_invocation('workflow-microservice', f"dapr-update-encounter/{encounter_id}", 'PUT', data)
    return response

def update_encounter_status(workflow_status_id: str, encounter_id: str):
    """
    This method invokes Workflow Microservice to update Encounter status
    :return: json
    """ 
    print("Invoke Workflow Microservice to update Encounter{} status to {}".format(encounter_id, workflow_status_id), flush=True)
    status_json_object = {}
    status_json_object["entityId"] = encounter_id
    response = invoke_microservice_using_service_invocation('workflow-microservice', f"dapr-update-encounter-status/{workflow_status_id}", 'PUT', status_json_object)
    return response

def update_medical_record_status(workflow_status_id: str, medical_record_id: str):
    """
    This method invokes Workflow Microservice to update MR status
    :return: json
    """ 
    print("Invoke Workflow Microservice to update MR {} status to {}".format(medical_record_id, workflow_status_id), flush=True)
    status_json_object = {}
    status_json_object["entityId"] = medical_record_id
    response = invoke_microservice_using_service_invocation('workflow-microservice', f"dapr-update-medical-record-status/{workflow_status_id}", 'PUT', status_json_object)
    return response

def dapr_get_workflow_id_for_label(label: str)->str:
    """
    This method invokes Workflow Microservice to get workflow id for label
    """ 
    print("Invoke Workflow Microservice to get workflow id for label \"{}\"".format(label), flush=True)
    json_response = invoke_microservice_using_service_invocation('workflow-microservice', 'dapr-get-workflow-id/{}'.format(label), 'GET')
    return json_response["workflow_object_id"]

def get_workflow_ids_for_labels():
    """
    This method invokes Workflow Microservice to get workflow ids for labels
    """ 
    print("Invoke Workflow Microservice to get workflow ids for labels", flush=True)
    workflow_ids_dict = dict()
    workflow_ids_dict['icd_extraction_in_progress'] = dapr_get_workflow_id_for_label("ICD Extraction in Progress")
    workflow_ids_dict['icd_extraction_complete'] = dapr_get_workflow_id_for_label("ICD Extraction Complete")
    workflow_ids_dict['final_review'] = dapr_get_workflow_id_for_label("Waiting for Final Review")
    return workflow_ids_dict

def get_icd_code_id(icd_code: str, year: int, plan: str)->str:
    print(f"get icd code {icd_code} ID for year: {year} and plan: {plan}", flush=True)
    label_id_request_object = dict()
    label_id_request_object["icdCode"] = icd_code
    # label_id_request_object["year"] = str(year) # deprecated
    label_id_request_object["date"] = datetime.datetime(year=year, month=1, day=1).isoformat()
    label_id_request_object["plan"] = plan
    json_response = invoke_microservice_using_service_invocation('workflow-microservice', "dapr-get-icd-label", 'POST', label_id_request_object)
    return json_response["Id"]

def get_encounter_boundaries(encounter_id):
    json_response = invoke_microservice_using_service_invocation('workflow-microservice', f'dapr-get-encounter-boundaries/{encounter_id}', 'GET')
    return json_response

def get_encounter_ocr_pages(encounter_id):
    json_response = invoke_microservice_using_service_invocation('workflow-microservice', f'dapr-get-ocr-pages-by-encounter-id/{encounter_id}', 'GET')
    return json_response

def save_icd_codes(encounter_id: str, extracted_icd_codes_result: dict):
    print(f"save icd code for encounter {encounter_id}",flush=True)
    json_response = invoke_microservice_using_service_invocation('workflow-microservice', f"dapr-save-icd-labels/{encounter_id}", 'PUT', extracted_icd_codes_result)
    return json_response

def update_microservice_state(state: str):
    """
    This method invokes Workflow Microservice to update microservice state
    :return: json
    """ 
    print("Invoke Workflow Microservice to update microservice state", flush=True)
    from constants import microservice
    health_state = dict()
    health_state["microservice"] = microservice
    health_state["state"] = state
    response = invoke_microservice_using_service_invocation(
        'workflow-microservice', 
        "dapr-update-microservice-health-state", 'PUT', health_state
    )
    return response