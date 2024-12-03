import numpy as np
import torch 
from typing import List, Union

class Token(object):
    def __init__(self, 
        token_id: int, 
        token_text: str,
        label: str = None,
        
        ) -> None:
        self.token_id = token_id 
        self.token_text = token_text
        self.label = label
        
    def __str__(self):
        return "Token(id={} text='{}' label = '{}')".format(self.token_id, self.token_text, self.label)
    def __repr__(self):
        return str(self)
        
    
    def is_part(self) -> bool:
        return self.token_text.startswith("##")


class SingleModelToken(Token):

    def __init__(self,
        token_id: int, 
        token_text: str,
        logits: Union[torch.Tensor, np.ndarray] = None,
        negation_prob = None,
        label: str = None
        ) -> None:

        super().__init__(token_id, token_text, label)

        if type(logits)!=np.ndarray:
            logits = np.aray(logits)
        self.probs = torch.softmax(torch.from_numpy(logits), dim=-1).numpy()
        self.logits = logits 
        self.negation_prob = negation_prob
        
        self.prediction = np.argmax(self.logits)
        self.confidence = self.probs[self.prediction]
 

class MultiModelToken(Token):

    def __init__(self,
        token_id: int, 
        token_text: str,
        logits: dict, # keys are the model names
        label: str = None
        )-> None:

        super().__init__(token_id, token_text, label)
        model_names = list(logits.keys())
        if type(logits[model_names[0]])!=np.ndarray:
            logits = {model_name: np.aray(logits[model_name]) for model_name in model_names}
        self.probs = {model_name: torch.softmax(torch.from_numpy(logits[model_name]), dim=-1).numpy() for model_name in model_names}
        self.logits = logits 
        
        self.predictions = {model_name: np.argmax(self.logits[model_name]) for model_name in self.logits}
        self.confidences = {model_name:self.probs[self.predictions[model_name]] for model_name in self.predictions}

    