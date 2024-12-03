import json
import numpy as np
import torch 
from torch import nn 
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm.auto import tqdm
from .tag_encoder import TagEncoder
import pickle
import re
from .preprocess import  tokenize_and_split_to_chunks, tokenizer, CLS_ID, SEP_ID, PAD_ID
from .chunk import SingleModelChunk
from .dataset import TestChunkDataset
import os
import IPython
from postprocess.utils import Aligner
def get_doc_predictions(resources, text, device):
    
    chunks_infos  = tokenize_and_split_to_chunks(text)

    model_names = ["medicare_commercial_model", "commercial_only_model", "medicare_only_model"]
    batch_size = 1
    doc_dataset = TestChunkDataset(chunks_infos)
    doc_loader = DataLoader(doc_dataset, batch_size = batch_size, drop_last=False, shuffle=False)
    outputs = {task_name:[] for task_name in model_names}
    
    for batch_index, batch in enumerate(doc_loader):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        negation_logits = resources["negation_model"](input_ids = input_ids, attention_mask = attention_mask).detach().cpu()
        for model_name in model_names:

            logits = resources[model_name]["model"](input_ids = input_ids, attention_mask = attention_mask)

            for i in range(len(input_ids)):            
                chunk = SingleModelChunk(
                            resources[model_name]["encoder"], 
                            input_ids[i].detach().cpu().numpy(), 
                            logits[i].detach().cpu().numpy(), 
                            negation_logits[i],
                            batch["labels_mask"][i].detach().cpu().numpy(), 
                            tokenizer, chunks_infos[batch_index * batch_size + i])
                outputs[model_name].append(chunk)
    return outputs
    
def get_doc_predicted_labels(doc_predictions, remove_negated):
    output = set()
    for chunk in doc_predictions:
        for entity in chunk.entities:
            if entity.negated and remove_negated:
                continue
            output.add(entity.prediction_label)
    if "O" in output:
        output.remove("O")
    return output

def split_composite(label):
    assert "_" in label, "The label is not composite"
    return label.split("_")

def should_include(larger_entity_text, texts):
    larger_entity_text = re.sub('[^0-9a-zA-Z]+', ' ', larger_entity_text).lower()
    for text in texts:
        text = re.sub('[^0-9a-zA-Z]+', ' ', text).lower()
        if text in larger_entity_text:
            return True 
    return False

def get_partial_overlapping_text(larger_entity_text, texts):
    original_entity_text = larger_entity_text
    larger_entity_text = re.sub('[^0-9a-zA-Z]+', ' ', larger_entity_text).lower()
    larger_text_found = None
    for text in texts:
        text = re.sub('[^0-9a-zA-Z]+', ' ', text).lower()
        if text in larger_entity_text:
            if larger_text_found is None or len(text) > len(larger_text_found):
                larger_text_found = text
    assert larger_text_found is not None
    assert larger_text_found  in larger_entity_text
    
    return larger_text_found 


def add_partial_overlapping_logic(predicted_entities, logic_dict):
    output = []
    for entity in predicted_entities:
        
        entity_text =  entity["text"].lower()
        entity_label = entity["label"].upper()
        
        if entity_label in logic_dict:
            for include_icd in logic_dict[entity_label]:
                if should_include(entity_text, logic_dict[entity_label][include_icd]):
                    

                    larger_overlapping_text = get_partial_overlapping_text(entity_text, logic_dict[entity_label][include_icd])
                    try:
                        start_index, end_index = get_partial_overlap_alignment_index(entity_text, larger_overlapping_text)
                    except:
                        continue
                    
                    output.append({
                        "start":start_index + entity["start"],
                        "end":end_index + entity["start"],
                        "label":include_icd,
                        "text":  entity_text[start_index: end_index]
                    })
    return output
def get_partial_overlap_alignment_index(larger_entity_text, smaller_entity_text):
    larger_entity_text = larger_entity_text.lower()
    smaller_entity_text  = smaller_entity_text.lower().strip()
    words = [word for word in re.split('([0-9,.?:;~!@#$%^&*()\s])', smaller_entity_text) if word.strip()!=""]
    word_index = 0
    current_index = 0
    start_index = None
    end_index = None
    while word_index < len(words) and current_index < len(larger_entity_text):
        while current_index < len(larger_entity_text) and not larger_entity_text[current_index:].startswith(words[word_index]):
            current_index += 1
        if larger_entity_text[current_index:].startswith(words[word_index]):
            if start_index is None:
                start_index = current_index
            current_index += len(words[word_index])
            end_index = current_index
            word_index  += 1
    if word_index == len(words):
        return (start_index, end_index)
    raise ValueError("Unable to find the index of {} inside: {}".format(larger_entity_text, smaller_entity_text))
    
def get_entity_tokens(entity):
    tokens = []
    for word in entity.words:
        for token in word.tokens:
            tokens.append(token.token_id)
    return tokens
def get_entity_last_word_index(entity_tokens, chunk_word_mapping, start_word_index):
    word_index = start_word_index
    current_word_tokens = [token for token in chunk_word_mapping[word_index]["tokens"] if token not in [CLS_ID, SEP_ID]]
    current_word_token_index = 0
    
    for index in range(len(entity_tokens)):
        e_token = entity_tokens[index]     
        
        if current_word_token_index < len(current_word_tokens):
            assert current_word_tokens[current_word_token_index] == e_token
            current_word_token_index += 1
        else:
            current_word_token_index = 0
            word_index += 1
            current_word_tokens = [token for token in chunk_word_mapping[word_index]["tokens"] if token not in [CLS_ID, SEP_ID]]
        

            assert current_word_tokens[current_word_token_index] == e_token
            current_word_token_index +=1
    
    return word_index
    
def get_chunk_entities_indices(chunk_word_mapping, entities):
    entity_start_word_index = 0
    entity_start_token_index = 0
    output = []
    index = 0
    for entity in entities:
        entity_tokens = get_entity_tokens(entity)
        word_tokens =  [token  for token in chunk_word_mapping[entity_start_word_index]["tokens"] if token not in [CLS_ID, SEP_ID]]
        index += 1
        assert entity_tokens[0] == word_tokens[0]
        entity_endword_index = get_entity_last_word_index(entity_tokens, chunk_word_mapping, entity_start_word_index)
        start_word = chunk_word_mapping[entity_start_word_index]
        end_word = chunk_word_mapping[entity_endword_index]
        output.append(
            {
                "entity":entity,
                "start": start_word["start"],
                "end":end_word["end"]
            }
        )
        entity_start_word_index = entity_endword_index + 1

    return output
def get_output_entity_indices(chunks, remove_negated):
    output = []
    for chunk_index, chunk in enumerate(chunks):
        entity_indices = get_chunk_entities_indices(chunk.word_token_mapping, chunk.entities)
         

        for entity_info in entity_indices:
            if entity_info["entity"].negated and remove_negated:
                continue
            entity_text = entity_info["entity"].text()
            
            output.append({
                "start":entity_info["start"],
                "end":entity_info["end"],
                "label": entity_info["entity"].prediction_label,
                "text":entity_text, 
            })
    return output

def alpha_equal(str1, str2):
    i = 0
    j = 0
    while i < len(str1) and j < len(str2):
        if str1[i].isalpha() and str2[j].isalpha():
            if str1[i] == str2[j]:
                i += 1
                j += 1
            else:
                return False
        elif str1[i].isalpha():
            j += 1
        elif str2[j].isalpha():
            i += 1
        else:
            i += 1
            j += 1
    return True


def get_models_predictions(resources, text, device, remove_negated=True):
    predictions_chunks = {}
    heirarchy_rule = resources["heirarchy_rule"]
    
    predictions_chunks = get_doc_predictions(resources, text, device)
    
    all_entities_data = []
    for category in predictions_chunks:
        entities_data = get_output_entity_indices(predictions_chunks[category], remove_negated = remove_negated)
        all_entities_data.extend(entities_data)
    overlapping_entities_data = add_partial_overlapping_logic(all_entities_data,resources["partial_overlapping_logic"])
    all_entities_data = all_entities_data + overlapping_entities_data
    output = []
    for entity_data in all_entities_data:
        if "_" in entity_data["label"]:
            for label in entity_data["label"].split("_"):
                output.append({
                    "start": entity_data["start"],
                    "end": entity_data["end"],
                    "label": label,
                    "text": entity_data["text"]
                })
        elif entity_data["label"]!="O":
            output.append({
                "start": entity_data["start"],
                "end": entity_data["end"],
                "label": entity_data["label"],
                "text": entity_data["text"]
            })
    
    output = heirarchy_rule.apply(output)
    
    return output
    
def compute_confusion_metrics(predictions, labels):
    predictions = set(predictions)
    labels = set(labels)
    tp = len(predictions & labels)
    fp = len(predictions - labels)
    fn = len(labels - predictions)
    
    return {
        "tp":tp,
        "fp":fp,
        "fn":fn
    }

def compute_iq_accuracy(tp, fp, fn):
    if (tp + fn) == 0:
        return 0
    output = (tp*100)/(tp+fn)-(30/(tp+fn))*fp
    if output < 0:
        output = 0
    return output




