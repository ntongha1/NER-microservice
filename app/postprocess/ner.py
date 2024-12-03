from utils import AtriusDocDataset
import json
from postprocess import PostRecord
import torch

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
class TagEncoder(object):
    def __init__(self, tag2index, index2tag, pad_label_id = None):
        self.tag2index = tag2index
        self.index2tag = {int(key):value for key, value in index2tag.items()}
        self.num_tags = len(self.tag2index)
        self.pad_label_id = pad_label_id
    def encode(self, item):
        return self.tag2index[item]
    def decode(self, index):
        return self.index2tag[index]
    def __str__(self):
        return str(self.tag2index)
    def __repr__(self):
        return str(self)
    


def get_phrase_tag_pairs(record, tokenizer):
    output = []
    for entity in record.predicted_entities:
        words = []
        tag = entity.get_tag()
        for word in entity.words:
            decoded_word = word.decode()
            words.append(decoded_word)
        
        output.append({"entity":" ".join(words), "tag":tag})
    return output

def get_doc_predictions(model, doc, encoder, tokenizer, device = device):
    doc_dataset = AtriusDocDataset(doc, encoder, tokenizer)
    records = []
    for index in range(len(doc_dataset)):
        inputs = doc_dataset[index]
        outputs = model(input_ids = inputs["input_ids"].unsqueeze(0).to(device), attention_mask = inputs["attention_mask"].unsqueeze(0).to(device)).detach().cpu()
        record = PostRecord(doc_dataset.records[index], tokenizer, encoder,  inputs["input_ids"], logits  = outputs[0], labels_mask = inputs["labels_mask"])
        records.append(record)
    return records