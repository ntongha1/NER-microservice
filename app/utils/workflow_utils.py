import re

from utils.dapr_utils import (
    get_icd_code_id,
    save_icd_codes
)


def get_icd_code_page_id_and_indices(visit_text: str, visit_start_index: int, icd_code_start_idx:int, icd_code_end_idx:int, no_of_visit_pages: int, ocr_page_ids: dict)->dict:
    print('get icd code page ID and indices', flush=True)
    icd_code_page_no_and_idx = dict()
    print(f'EasyOCRPageDividerStart index: {visit_text[:icd_code_start_idx].rfind("EasyOCRPageDividerStart")}', flush=True)
    print(f'EasyOCRPageDividerEnd index: {visit_text[:icd_code_start_idx].rfind("EasyOCRPageDividerEnd")}', flush=True)
    if no_of_visit_pages == 1:
        icd_code_page_no_and_idx['page_id'] = ocr_page_ids[str(0)]
        icd_code_page_no_and_idx['start_idx'] = icd_code_start_idx
        icd_code_page_no_and_idx['end_idx'] = icd_code_end_idx + 1
    elif visit_text[:icd_code_start_idx].rfind("EasyOCRPageDividerStart") != -1:
        icd_code_start_page_divider_idx = visit_text[:icd_code_start_idx].rfind("EasyOCRPageDividerStart")
        icd_code_start_page_divider_len_idx = icd_code_start_page_divider_idx + len("EasyOCRPageDividerStart")
        # print(f"icd_code_page_divider_idx {icd_code_start_page_divider_idx}")
        icd_code_page_no = int(re.search(r'\d+', visit_text[icd_code_start_page_divider_len_idx:int(visit_text.find('\n\n', icd_code_start_page_divider_len_idx))]).group(0))-1
        print(f"icd code page no: {icd_code_page_no}", flush=True)
        icd_code_page_no_and_idx['page_id'] = ocr_page_ids[str(icd_code_page_no)]
        icd_code_page_no_and_idx['start_idx'] = len(visit_text[icd_code_start_page_divider_idx:icd_code_start_idx])
        icd_code_page_no_and_idx['end_idx'] = icd_code_page_no_and_idx['start_idx'] + (icd_code_end_idx - icd_code_start_idx) + 1
    else:
        icd_code_page_no_and_idx['page_id'] = list(ocr_page_ids.values())[0]
        icd_code_page_no_and_idx['start_idx'] = visit_start_index + icd_code_start_idx 
        icd_code_page_no_and_idx['end_idx'] = icd_code_page_no_and_idx['start_idx'] + (icd_code_end_idx - icd_code_start_idx) + 1
    
    return icd_code_page_no_and_idx


def create_icd_code_location_object(ocr_page_id: str, start_idx: int, end_idx: int, chunk: str)->dict:
    icd_location_object =  dict()
    icd_location_object["pageId"] = ocr_page_id
    icd_location_object["startIdx"] = start_idx
    icd_location_object["endIdx"] = end_idx
    icd_location_object["value"] = chunk
    icd_location_object["createdBy"] = "icd model"

    return icd_location_object


def create_extracted_icd_codes_locations_object(icd_codes_locations: dict, icd_code: str, chunk: str, ocr_page_id: str, start_idx: int, end_idx: int)->dict:
    icd_location_object = create_icd_code_location_object(ocr_page_id, start_idx, end_idx, chunk)
    if not icd_code in icd_codes_locations:
        icd_codes_locations[icd_code] = list()
    icd_codes_locations[icd_code].append(icd_location_object)

    return icd_codes_locations


def create_extracted_icd_codes_result_object(icd_code: str, chunk: str, ocr_page_id: str, start_idx: int, end_idx: int, year: int, plan: str)->dict:
    icd_code_id = get_icd_code_id(icd_code, year, plan)
    icd_location_object = create_icd_code_location_object(ocr_page_id, start_idx, end_idx, chunk)
    icd_label_locations = dict()
    icd_label_locations["labelId"] = icd_code_id
    icd_label_locations["locations"] = icd_location_object
    icd_extraction_result = dict()
    icd_extraction_result["service"] = "ICD"
    icd_extraction_result["labels"] = list()
    icd_extraction_result["labels"].append(icd_label_locations)

    return icd_extraction_result


def save_extracted_icd_codes_location(encounter_id: str, icd_code: str, year: int, plan: str, icd_codes_locations: list)->dict:
    icd_code_id = get_icd_code_id(icd_code, year, plan)
    icd_label_locations = dict()
    icd_label_locations["labelId"] = icd_code_id
    icd_label_locations["locations"] = icd_codes_locations

    icd_extraction_result = dict()
    icd_extraction_result["service"] = "ICD"
    icd_extraction_result["labels"] = list()
    icd_extraction_result["labels"].append(icd_label_locations)
    response = save_icd_codes(encounter_id, icd_extraction_result)

    return response
