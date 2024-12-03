import json 
import torch 
from transformers import AutoTokenizer
from .ner import get_doc_predictions, TagEncoder
import tqdm


def get_disease_entities_from_records(records, tokenizer):
    output = list()
    for record in records:
        for entity in record.predicted_entities:
            tag = entity.get_tag()
            if tag != 'O':
                output.append(" ".join(word.decode(tokenizer) for word in entity.words))
    return output

def get_disease_ners_e2e_dataset(data):
    with open("resources/disease_ner_tag_mapping.json") as input_file:
        encoder_data = json.load(input_file)
    encoder = TagEncoder(pad_label_id = -100, **encoder_data)
    
    model = torch.load("resources/disease-ner-model.pt", map_location=torch.device('cpu'))
    
    tokenizer = AutoTokenizer.from_pretrained("emilyalsentzer/Bio_ClinicalBERT")
    
    doc_predictions = []
    for doc in tqdm.auto.tqdm(data):
        entities = get_disease_entities_from_text_fast(model, doc["data"]["text"], encoder, tokenizer)
        doc_predictions.append(dict(text=doc["data"], entities=entities))
    return doc_predictions

def get_disease_entities_from_text_fast(model, text, encoder, tokenizer):
    records = get_doc_predictions(model, {"text":text}, encoder, tokenizer)
    entities = get_disease_entities_from_records(records, tokenizer)
    return entities

def get_diease_entities_from_text_slow(text):
    with open("resources/disease_ner_tag_mapping.json") as input_file:
        encoder_data = json.load(input_file)
    encoder = TagEncoder(pad_label_id = -100, **encoder_data)
    
    model = torch.load("resources/disease-ner-model.pt", map_location=torch.device('cpu'))
    
    tokenizer = AutoTokenizer.from_pretrained("emilyalsentzer/Bio_ClinicalBERT")
    records = get_doc_predictions(model, {"text":text}, encoder, tokenizer)
    entities = get_disease_entities_from_records(records, tokenizer)
    return entities