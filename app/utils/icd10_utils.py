from torch.utils.data import Dataset, DataLoader
import tqdm
import pandas as pd
import torch
import IPython
import os
import re 

def prepare_sentence(tokenizer, record, encoder, max_length):
    text = record
    

    input_ids = tokenizer.encode(text)
    output_labels_mask = [1 for i in range(len(input_ids))]
    output_labels_mask[-1] = 0
    output_labels_mask[0] = 0
    

    mask = [1]  * len(input_ids)
    
    while len(input_ids) < max_length:
        input_ids.extend(tokenizer.convert_tokens_to_ids(["[PAD]"]))
        output_labels_mask.append(0)
        mask.append(0)
    
    if len(input_ids) > max_length:
        input_ids = input_ids[:max_length]
        mask = mask[:max_length]
        output_labels_mask = output_labels_mask[:max_length]
       
        
     
    
    return {
        "input_ids":torch.tensor(input_ids),
        "attention_mask":torch.tensor(mask),
       
        "labels_mask":torch.tensor(output_labels_mask)
    }        



class AtriusDocDataset(Dataset):
    """The class encapsulate Atrius dataset using the initial preprocessing.
    The initial preprocessing doesn't use combo code. 
    """
    def __init__(self, doc_data, encoder, tokenizer, max_length = 512, max_words_per_chunk = 150):
        super().__init__()
        
        self.records = []
        self.encoder = encoder
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.max_words_per_chunk = max_words_per_chunk
        self.records = self.split_to_pragraphs(doc_data)
      
    def __len__(self):
        return len(self.records)
    def __getitem__(self, index):
        record = self.records[index]
        return prepare_sentence(self.tokenizer, record, self.encoder, self.max_length)
        
    
            
            
    def split_to_pragraphs(self, doc_data):
        """
        The document might contain multiple paragraph. This function will be used
        
        to split the document into those paragraphs while maintaining the label alignments.
        
        
        """
        paragraphs = re.split("\n", doc_data["text"])
        paragraphs = self.merge_possible(paragraphs)
        return paragraphs
                    
                    
    def paragraphs_length(self, paragraphs):
        text = " ".join(paragraphs)
        return len(text.split())
    def merge_paragraphs(self, paragraphs):
        output = ""
        for paragraph in paragraphs:
            output += paragraph + "\n"
        return output

    def merge_possible(self, paragraphs):
        output = []
        index = 0
        paragraphs_to_merge = []
        while index < len(paragraphs):
            if self.paragraphs_length(paragraphs_to_merge) > self.max_words_per_chunk or self.paragraphs_length(paragraphs_to_merge + [paragraphs[index]]) > self.max_words_per_chunk:
                output.append(self.merge_paragraphs(paragraphs_to_merge))
                paragraphs_to_merge = [paragraphs[index]]
            else:
                
                paragraphs_to_merge.append(paragraphs[index])
            index += 1
        if len(paragraphs_to_merge) > 0:
            output.append(self.merge_paragraphs(paragraphs_to_merge))
        return output
