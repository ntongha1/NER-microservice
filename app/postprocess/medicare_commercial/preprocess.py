
from datetime import datetime
from telnetlib import IP
from tqdm.auto import tqdm
import re
import time
from transformers import AutoTokenizer
from tqdm.auto import tqdm
import string
import IPython
punctuations = set(string.punctuation)

tokenizer = AutoTokenizer.from_pretrained("emilyalsentzer/Bio_ClinicalBERT")
CLS_ID, SEP_ID, PAD_ID = tokenizer.convert_tokens_to_ids(["[CLS]", "[SEP]", "[PAD]"])






def tokenize_sentence(text, sentence):
    sent_text = text[sentence["begin"]:sentence["end"]+1]
    words = get_words(sent_text)
    for word_info in words:
        tokens = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(word_info["text"]))
        word_info["tokens"] = tokens
        word_info["start"] += sentence["begin"]
        word_info["end"] += sentence["begin"]
    words[-1]["tokens"].append(SEP_ID)
    return words
def get_num_tokens(word_tokens):
    num_tokens = 0
    for word_info in word_tokens:
        num_tokens += len(word_info["tokens"])
    return num_tokens
def get_document_chunks(document_sentences_tokens, max_tokens_per_chunk = 512):
    
    output = []
    for sent_word_tokens in document_sentences_tokens:
        
        if len(output) == 0:
            output.append([word_info for word_info in sent_word_tokens])
        else:
            prev_chunk = output[-1]
            if get_num_tokens(prev_chunk) + get_num_tokens(sent_word_tokens) > max_tokens_per_chunk:
                output.append([word_info for word_info in sent_word_tokens])
            else:
                prev_chunk.extend([word_info for word_info in sent_word_tokens])
    return output
        
def get_words(string):
    output = []
    current_index = 0
    start_index = current_index
    while current_index < len(string):
        if string[current_index].isspace():
            if start_index == current_index:
                current_index += 1
                start_index = current_index
            else:
                output.append({
                    "text":string[start_index: current_index],
                    "start":start_index,
                    "end":current_index
                })
                current_index += 1
                start_index = current_index
        elif string[current_index] in punctuations:
            if start_index < current_index:
                output.append({
                        "text":string[start_index: current_index],
                        "start":start_index,
                        "end":current_index
                })

            output.append({
                    "text":string[current_index:current_index + 1],
                    "start":current_index,
                    "end":current_index + 1
                })
            
            current_index += 1
            start_index = current_index
        elif string[current_index].isdigit():
            if start_index < current_index:
                output.append({
                        "text":string[start_index: current_index],
                        "start":start_index,
                        "end":current_index
                })
            num_start_index = current_index 
            while current_index < len(string) and string[current_index].isdigit():
                current_index += 1
            output.append({
                    "text":string[num_start_index: current_index],
                    "start":start_index,
                    "end":current_index
            })
            start_index = current_index
        else:
            current_index +=  1
    if start_index!=current_index and not string[start_index:current_index].isspace():
        output.append({
                    "text":string[start_index: current_index],
                    "start":start_index,
                    "end":current_index
                })
    
    return output
def extract_sentences(text):

    output = []
    sentences = [sent.strip() for sent in re.split("[\n\n]", text) if len(sent.strip()) > 0]
    current_index = 0
    for sentence in sentences:
        while text[current_index].isspace():
            current_index += 1

        assert text[current_index:].startswith(sentence)
        output.append(
            {
                "begin":current_index,
                "end": current_index + len(sentence)
            }
        )
        current_index += len(sentence)
    return output




def tokenize_document(text):
    sentences = extract_sentences(text)
    output = []
    
    for sentence in sentences:
        sent_word_tokens = tokenize_sentence(text, sentence)
        output.append(sent_word_tokens)
    return output
def tokenize_and_split_to_chunks(text):
    document_words_token_mapping = tokenize_document(text)
    return get_document_chunks(document_words_token_mapping)