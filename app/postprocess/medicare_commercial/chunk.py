from .token import  Token, SingleModelToken, MultiModelToken
from .word import Word, SingleModelWord, MultiModelWord
from .entity import Entity, SingleModelEntity
import IPython
import os
import torch
class Chunk(object):
    def __init__(self):
        pass

class SingleModelChunk(Chunk):
    def __init__(self, tag_encoder, input_ids, logits, negation_logits, labels_mask, tokenizer, word_token_mapping):
        super().__init__()
        self.tag_encoder = tag_encoder
        self.tokens = self.build_tokens(input_ids, logits, negation_logits, labels_mask, tokenizer)
        self.words = self.build_words(self.tokens)
        self.entities = self.build_entities(self.words)
        self.word_token_mapping = word_token_mapping
    def build_entities(self, words):
        entities = []
        index = 0
        for word in words:
            
            index += 1
            if len(entities) == 0:
                entities.append(Entity([word]))
            else:
                prev_entity = entities[-1]
                if prev_entity.prediction_label == "O" and word.prediction_label == prev_entity.prediction_label:
                    prev_entity.append_word(word)
                    prev_entity.build_entity()
                elif word.prediction_label.startswith("I-"):
                    if  word.prediction_label[2:] == prev_entity.prediction_label:
                        prev_entity.append_word(word)
                        prev_entity.build_entity()
                    else:
                        entities.append(Entity([word]))
                else:
                    entities.append(Entity([word]))
        self.entities = entities
        return entities

                    

    def build_words(self, tokens):
        words = []
        for token in tokens:
            if len(words) == 0:
                words.append(SingleModelWord([token], self.tag_encoder))
            elif token.is_part():
                last_word = words[-1]
                last_word.append_token(token)
                last_word.build_text()
            else:
                words.append(SingleModelWord([token], self.tag_encoder))
        self.words = words
        return words
    def build_tokens(self, input_ids, logits, negation_logits, labels_mask, tokenizer):
        
        negation_probs = torch.softmax(negation_logits, dim=-1).detach()[:,  -1].numpy()
        input_ids = [input_id  for input_id, label_mask in zip(input_ids, labels_mask) if label_mask == 1]
        logits = [logit for logit, label_mask in zip(logits, labels_mask) if label_mask == 1]
        negation_logits = [logit for logit, label_mask in zip(negation_probs, labels_mask) if label_mask == 1]

        assert(len(input_ids) == len(logits))
        output = []
        for i in range(len(input_ids)):
            token_id = input_ids[i]
            token_text = tokenizer.decode(token_id)
            token_logits = logits[i]
            token_negation_prob = negation_logits[i]
            token = SingleModelToken(token_id = token_id, token_text=token_text, logits=token_logits, negation_prob=token_negation_prob)
            output.append(token)
        self.tokens = output
        return output


