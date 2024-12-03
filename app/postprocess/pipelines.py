from transformers import AutoTokenizer
import json
from .ner import TagEncoder
from .ner import get_doc_predictions
from .utils import Aligner
import torch
import pickle

from .medicare_commercial.tag_encoder import load_tag_encoder
from .medicare_commercial.models import load_medicare_commercial_model
from .medicare_commercial.util import get_models_predictions
from postprocess.rules import HeirarchyRule, load_rule

def get_icd10_codes_with_indices_from_text_fast(model, text, encoder, tokenizer):
    records = get_doc_predictions(model, {"text":text}, encoder, tokenizer)
    
    output_entities = []
    
    text = text.lower()
    
    aligner = Aligner(text)
    error_occured = False
    current_text_index = 0
    for record in records:
        record_index =  text[current_text_index:].index(record.text[:50].lower().strip())
        current_text_index += record_index 
        
        for entity in record.predicted_entities:
            
            entity_text = " ".join(word.decode(tokenizer) for word in entity.words)
            if entity_text=="[UNK]":
                print("[UNK] found in '{}'".format(text[current_text_index: current_text_index + 100].lower().strip()) )
                # Case: xylocainewithepinephrineforlocalanesthesiaandsterileprepbiopsyperformedwitha15scalpelbladelightelect
                break
            try:
                start, end = aligner.get_alignment_indices(entity_text, current_text_index)
                output_entities.append({"start":start, "end": end, "text":text[start:end], "icd10_code":entity.get_tag()})
                current_text_index = end 
            except:
                error_occured = True
                break 
        if error_occured:
            break
            
    return  output_entities
class Pipeline(object):
    def __init__(self, config):
        self.tokenizer =  AutoTokenizer.from_pretrained("emilyalsentzer/Bio_ClinicalBERT")
        
        self.initialize_from_config(config)
    def initialize_from_config(self, config):
        raise NotImplementedError("Not implemented yet!")

    def __call__(self, text):
        raise NotImplementedError("Not implemented yet!")

class OldDeploymentPipeline(Pipeline):
    """A class to wrap the old deployment pipeline
    """

    def __init__(self, config):
        """
        Initialize the class object using the config data.
        The config has the following format

        {
            "model_path": "/path/to/the/model",
            "tag_map_path":"/path/to/the/tag/encoder/json/file"
        }

        """
        super().__init__(config)
    def __call__(self, text):
        output_entities = get_icd10_codes_with_indices_from_text_fast(self.model, text, self.tag_encoder, self.tokenizer)
        return output_entities
    def initialize_from_config(self, config):
        self.model = self.load_model_using_config(config)
        self.tag_encoder = self.load_tag_encoder_using_config(config)
    def load_tag_encoder_using_config(self, config):
        with open(config["tag_map_path"]) as input_file:
            tag_encoder_data = json.load(input_file)
        tag_encoder = TagEncoder(pad_label_id = -100, **tag_encoder_data)
        return tag_encoder
    def load_model_using_config(self, config):
        state = torch.load(config["model_path"], map_location=torch.device('cpu'))
        model = state['model']
        return model

class MedicareCommercialMultiModelPipeline(Pipeline):
    def __init__(self, config):
        super().__init__(config)
        
    def __call__(self, text):
        output_entities = get_models_predictions(self.resources, text, device = torch.device('cpu'))
        for ent in output_entities:
            ent["icd10_code"] = ent["label"]
        return output_entities
    def initialize_from_config(self, config):
        self.resources = self.load_resources_using_config(config)
    
    def load_resources_using_config(self, config):
        resources = {}
        model_names = ["medicare_commercial_model", "commercial_only_model", "medicare_only_model"]

        for model_name in model_names:
            model = load_medicare_commercial_model(config[model_name]["model_path"])
            model.eval()
            tag_encoder = load_tag_encoder(config[model_name]["tag_map_path"])
            resources[model_name] = {"model":model, "encoder":tag_encoder}
        with open(config["partial_overlapping_logic_data_path"]) as input_file:
            resources["partial_overlapping_logic"] = json.load(input_file)
        return resources

class FusedMedicareCommercialModelPipeline(MedicareCommercialMultiModelPipeline):
    def load_resources_using_config(self, config):
        resources = {"encoders":{}}
        model_names = ["medicare_commercial", "commercial_only", "medicare_only"]

        for model_name in model_names:
            tag_encoder = load_tag_encoder(config[model_name]["tag_map_path"])
            resources["encoders"][model_name] = tag_encoder
        model = load_medicare_commercial_model(config["model_path"])
        model.eval()
        resources["model"] = model
        with open(config["partial_overlapping_logic_data_path"]) as input_file:
            resources["partial_overlapping_logic"] = json.load(input_file)
        resources["heirarchy_rule"] = load_rule(config["heirarchy_rule_path"])
        
        return resources


class JointModelPipeline(MedicareCommercialMultiModelPipeline):
    def load_resources_using_config(self, config):
        resources = {}

        
        resources["encoders"] = load_tag_encoder(config["tag_encoders_path"])
       
       
        joint_model = load_medicare_commercial_model(config["joint_model_path"])
        joint_model.eval()
        resources["joint_model"] = joint_model


        negation_model = load_medicare_commercial_model(config["negation_model_path"])
        negation_model.eval()
        resources["negation_model"] = negation_model
        
        with open(config["partial_overlapping_logic_data_path"]) as input_file:
            resources["partial_overlapping_logic"] = json.load(input_file)
        resources["heirarchy_rule"] = load_rule(config["heirarchy_rule_path"])
        
        return resources


class SplitModelPipeline(MedicareCommercialMultiModelPipeline):
    def load_resources_using_config(self, config):
        resources = super().load_resources_using_config(config)

       

        negation_model = load_medicare_commercial_model(config["negation_model_path"])
        negation_model.eval()
        resources["negation_model"] = negation_model
        
        
        resources["heirarchy_rule"] = load_rule(config["heirarchy_rule_path"])
        
        return resources