import numpy as np 
import torch 
from spacy import displacy
import unicodedata
import string

class Aligner(object):
    def __init__(self, text):
        self.text = text.lower()
    def get_alignment_indices(self, entity_text:str, start_index:int = 0) -> tuple:
        """Get aligned position of the entity text inside the text.
        It is assumed that the entity_text will be inside the text. But
        the enity text might exist in extact form. Sometimes white spaces and
        other invisible characters might be added to the both the original text 
        and to the entity text.

        Args:
            entity_text (str): Entity text to align
            start_index (int): Start position to look inside the text. Default 0

        Returns:
            tuple: Start and end position of the entity text
        """
        
        entity_text = entity_text.lower()
        
        start = None 
        
        end = start_index
        entity_index = 0
        
        
        while end < len(self.text) and entity_index < len(entity_text):
            if self.text[end] == entity_text[entity_index]:
                if start is None:
                    start = end
                end += 1
                entity_index += 1
            elif list(unicodedata.normalize("NFD", self.text[end]))[0] == entity_text[entity_index]:
                if start is None:
                    start = end
                end += 1
                entity_index += 1
            elif ord(self.text[end]) > 128 and ord(entity_text[entity_index]) > 128:
                end += 1
                entity_index += 1
            elif ord(self.text[end]) > 128:
                end += 1
            elif ord(entity_text[entity_index]) > 128:
                entity_index += 1
            elif self.text[end].isspace():
                end += 1
            elif entity_text[entity_index].isspace():
                entity_index += 1
            else:
                
                print("Unexpected:'{}' inside: '{}'".format(entity_text, self.text[start_index:start_index + 100]))
                raise ValueError("Unexpected:'{}' inside: '{}'".format(entity_text, self.text[start_index:start_index + 100]))
            
        return start, end
        

class Token(object):
    def __init__(self, token_id, token_text, probs = None, label = None):
        super().__init__()
        self.token_id = token_id
        self.token_text = token_text
        self.probs = probs
        self.label = label
        
    def __repr__(self):
        return "Token(id = {}, text= {}, probs = {} label = {})".format(self.token_id, self.token_text, self.probs, self.label)
    def __str__(self):
        return repr(self)
    def decode(self, tokenizer):
        return tokenizer.decode([self.token_id])[0]
    def is_part(self):
        return self.token_text.startswith("##")
 
class Word(object):
    def __init__(self, tokens, tag_encoder):
        super().__init__()
        self.tokens = tokens
        self.tag_encoder = tag_encoder
        
    def __repr__(self):
        return "Word(tokens = {})".format(self.tokens)
    def __str__(self):
        return repr(self)
    def decode(self, tokenizer):
        return tokenizer.decode([token.token_id for token in self.tokens])
    def compute_probs(self):
        output = np.array([token.probs.numpy() for token in self.tokens])
        output = output.mean(axis=0)
        return output
    def get_bio_tag(self):
        probs = self.compute_probs()
        class_ = probs.argmax(axis=-1)
        return self.tag_encoder.decode(class_)
    def get_tag(self):
        bio_tag = self.get_bio_tag()
        if bio_tag == "O":
            return "O"
        return bio_tag[2:]
    def get_label(self):
        label = self.tag_encoder.decode(self.tokens[0].label)
        if label != "O":
            label = label[2:]
        for token in self.tokens:
            assert token.label is not None, "get_label works when labels for each token is available"
            current_label = self.tag_encoder.decode(self.tokens[0].label)
            if current_label != "O":
                current_label = current_label[2:]
            assert current_label == label, "Labels of all tokens should be the same"
        return label
class Entity(object):
    def __init__(self, words, tag_encoder):
        super().__init__()
        self.tag_encoder = tag_encoder
        self.words = words
        first_word_tag = self.words[0].get_tag()
        for word in self.words:
            assert word.get_tag() == first_word_tag
        
    def __repr__(self):
        return "Entity(words = {})".format(self.words)
    def __str__(self):
        return repr(self)
    def get_tag(self):
        return self.words[0].get_tag()
    def get_label(self):
        return self.words[0].get_label()


class PostRecord(object):
    def __init__(self, text, tokenizer, tag_encoder,  input_ids, logits  = None, predictions = None, labels = None, labels_mask = None, pad_label_index = -100):
        super().__init__()
        assert logits is not None or predictions is not None, "Provide either logits or predictions"
        assert not(logits is not None and predictions is not None), "Only one of the logits or predictions should be given"
        self.text = text
        self.tokenizer = tokenizer
        self.tag_encoder = tag_encoder
        self.pad_label_index = pad_label_index
        
        if logits is not None:
            predictions = torch.softmax(logits, dim=-1)
        assert len(input_ids) == len(predictions)
        if labels is not None:
            assert len(input_ids) == len(labels)
    
        if labels_mask is not None:
            assert len(input_ids) == len(labels_mask)
        self.tokens = self.build_tokens(input_ids, predictions, labels, labels_mask)
        self.words = self.build_words(self.tokens)
        self.predicted_entities = self.build_entities(self.words)
        if labels is not None:
            self.actual_entities = self.build_actual_entities(self.words)
        else:
            self.actual_entities = None
    def build_actual_entities(self, words):
        entities = []
        for word in words:
        
            if len(entities) > 0 and word.get_label().startswith("I-") and \
                word.get_label() == entities[-1].words[-1].get_label():
                entities[-1].words.append(word)
            elif word.get_label().startswith("I-"):
                entities.append(Entity([word], self.tag_encoder))
                
            else:
                entities.append(Entity([word], self.tag_encoder))
        return entities
    
        
        
    def build_tokens(self, input_ids, predictions, labels, labels_mask):
        tokens = []
        for i in range(len(input_ids)):
            _id = input_ids[i]
            probs = predictions[i]
            if labels is not None:
                label = labels[i]
            else:
                label = None
            if labels_mask is not None and labels_mask[i] == 0:
                continue
            
            if label == self.pad_label_index:
                continue
            text = self.tokenizer.decode([_id])
            token = Token(_id,  text, probs, label)
            tokens.append(token)
        return tokens
    def build_words(self, tokens):
        words = []
        for token in tokens:
            if token.is_part():
                words[-1].tokens.append(token)
            else:
                words.append(Word([token], self.tag_encoder))
        return words
    def build_entities(self, words):
        entities = []
        for word in words:
            if len(entities) > 0 and word.get_bio_tag().startswith("I-") and word.get_tag() == entities[-1].words[-1].get_tag():
                entities[-1].words.append(word)
            elif word.get_bio_tag().startswith("I-"):
                entities.append(Entity([word], self.tag_encoder))
                
            else:
                entities.append(Entity([word], self.tag_encoder))
        return entities
    def get_predicted_phrase_tag_pairs(self):
        output = []
        for entity in self.predicted_entities:
            phrase = " ".join([word.decode(self.tokenizer) for word in entity.words])
            tag = entity.get_tag()
            
            output.append(dict(phrase = phrase, tag = tag))
        return output
    def get_actual_phrase_tag_pairs(self):
        output = []
        for entity in self.actual_entities:
            phrase = " ".join([word.decode(self.tokenizer) for word in entity.words])
            tag = entity.get_label()
            
            output.append(dict(phrase = phrase, tag = tag))
        return output

                
        
            
        
        
class SpacyVisualizer(object):
    def __init__(self, nlp, options = {}):
        self.nlp = nlp 
        self.options = options
    def visualize(self, phrase_tags):
        
        output_text = ""
        output_tags = []
        for phrase_tag in phrase_tags:

            start = len(output_text)
            end = start + len(phrase_tag["phrase"])
            output_text += phrase_tag["phrase"]

            output_tags.append([start, end, phrase_tag["tag"]])
            output_text += " "
        output_text += "\n"
        
        doc = self.nlp.make_doc(output_text)    

        ents = []
        for span_start, span_end, label in output_tags:
            if label == "O" or label is None:
                continue
            ent = doc.char_span(span_start, span_end, label=label)
            if ent is None:
                continue
            ents.append(ent)

        doc.ents = ents
        options = {"distance":140}
        options.update(self.options)
        displacy.render(doc, style="ent", jupyter=True, options=options)
        return doc