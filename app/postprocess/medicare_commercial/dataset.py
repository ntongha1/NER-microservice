from torch.utils.data import Dataset, DataLoader
import numpy as np
import torch 
from .preprocess import CLS_ID, SEP_ID, PAD_ID
import IPython

class TestChunkDataset(Dataset):
    def __init__(self, chunks, max_tokens_per_chunk = 512):
        self.inputs = []
        self.labels_masks = []
        self.attention_masks = []
        for chunk in chunks:
            input_ids = []
            chunk[0]["tokens"].insert(0, CLS_ID)
            for word_info in chunk:
                input_ids.extend(word_info["tokens"])

          
            chunk_labels_mask = [1 if id not in {CLS_ID, SEP_ID, PAD_ID} else 0 for id in input_ids]
                
            chunk_labels_mask = chunk_labels_mask


            attention_mask = [1] * len(input_ids)

            while len(input_ids) < max_tokens_per_chunk:
                input_ids.append(PAD_ID)
                chunk_labels_mask.append(0)
                attention_mask.append(0)
            input_ids = input_ids[:max_tokens_per_chunk]
            chunk_labels_mask = chunk_labels_mask[:max_tokens_per_chunk]
            attention_mask = attention_mask[:max_tokens_per_chunk]

            self.inputs.append(input_ids)
            self.labels_masks.append(chunk_labels_mask)
            self.attention_masks.append(attention_mask)
    def __len__(self):
        return len(self.inputs)
    def __getitem__(self, index):
        input_ids = self.inputs[index]
        labels_mask = self.labels_masks[index]
        attention_mask = self.attention_masks[index]

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "labels_mask": torch.tensor(labels_mask, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.float32),
        }
