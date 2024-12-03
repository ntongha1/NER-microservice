import numpy as np
from typing import List, Union
from .token import Token, SingleModelToken, MultiModelToken
from .tag_encoder import TagEncoder
from collections import defaultdict

class Word(object):
    def __init__(self, tokens: List[Token]):
        self.tokens = tokens
    
    def build_text(self) -> str:
        output = ""
        first = True
        for token in self.tokens:
            if first:
                assert not token.token_text.startswith("##")
                output += token.token_text
                first = False 
            else:
                assert token.token_text.startswith("##")
                output += token.token_text[2:]
        return output
    def text(self):
        return self.build_text()
    

class SingleModelWord(Word):
    def __init__(self, tokens: List[SingleModelToken], tag_encoder: TagEncoder):
        super().__init__(tokens)
        self.tag_encoder = tag_encoder
        self.build_label()

    def build_label(self):
        logits = np.array([token.logits for token in self.tokens])
        self.negation_prob = np.mean([token.negation_prob for token in self.tokens])
        first = True 
        self.is_start_of_entity = False
        token_class_probs = defaultdict(list)
        for token in self.tokens:
            token_class = self.tag_encoder.decode(token.prediction)
            if first:
                first = False
                if token_class.startswith("B-"):
                    self.is_start_of_entity = True

            if token_class=="O":
                token_class_probs[token_class].append(token.confidence)
            else:
                token_class_probs[token_class[2:]].append(token.confidence)
        token_class_probs_avg = {}
        for token_class in token_class_probs:
            token_class_probs_avg[token_class] = np.mean(token_class_probs[token_class])
        max_class = None 
        max_confidence = None 
        for token_class in token_class_probs_avg:
            if max_class is None or max_confidence < token_class_probs_avg[token_class]:
                max_class = token_class 
                max_confidence = token_class_probs_avg[token_class] 
        
        self.logits = np.mean(logits, axis=0)


        if max_class == "O":
            self.prediction_label = max_class
        else:
            if self.is_start_of_entity:
                self.prediction_label = "B-"+max_class
            else:
                self.prediction_label = "I-"+max_class
        self.confidence = max_confidence
    

    def append_token(self, token):
        self.tokens.append(token)
        




class MultiModelWord(Word):
    def __init__(self, tokens: List[MultiModelToken], tag_encoders: TagEncoder):
        super().__init__(tokens)
        model_names = list(tokens[0].logits.keys())
        logits = {model_name: np.array([token.logits[model_name] for token in self.tokens]) for model_name in model_names}
        self.logits = {model_name: np.mean(logits[model_name], axis=0) for model_name in model_names}
        self.predictions = {model_name: np.argmax(self.logits[model_name], axis = -1) for model_name in model_names}
        self.tag_encoders = tag_encoders
        
        
        self.prediction_labels = {model_name: self.tag_encoders[model_name].decode(self.predictions[model_name])for model_name in model_names}
    

        



        