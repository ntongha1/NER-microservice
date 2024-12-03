import torch 
import numpy as np 


class Entity(object):
    def __init__(self, words):
        self.words = words
        self.build_entity()
    def append_word(self, word):
        self.words.append(word)
        self.build_entity()
        
    def build_entity(self):
        self.is_icd10 = False
        words_texts = []
        first_word = self.words[0]
        if first_word.prediction_label == "O":
            self.prediction_label = first_word.prediction_label
        else:
            self.prediction_label = first_word.prediction_label[2:]
        for word in self.words:
            if word.prediction_label!="O":
                self.is_icd10 = True 
            if self.prediction_label == "O":
                assert word.prediction_label == "O"
            else:
                assert word.prediction_label[2:] == self.prediction_label
            
            words_texts.append(word.text())
        text = " ".join(words_texts)
        self.negation_prob = np.mean([word.negation_prob for word in self.words])
        self.negated = self.negation_prob > 0.1
        return text
    def text(self):
        return self.build_entity()

class SingleModelEntity(Entity):
    def __init__(self, words):
        super().__init__(words)


