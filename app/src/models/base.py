
import torch 
from torch import nn 
from torch.nn import functional as F
from transformers import AutoModel



class Classifier(nn.Module):
    def __init__(self, embedding_dim, num_classes, hidden_dim = 256, prototype_embeddings_dim = 8,  dropout=0.25):
        super().__init__()
        self.dropout = dropout
        
        self.linear = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(p = dropout),
            nn.Linear(hidden_dim, prototype_embeddings_dim),

        )
        
        
        
        # Prototypes for each classes
        self.prottypes = nn.Linear(num_classes, prototype_embeddings_dim)


    
    def forward(self, inputs, return_embeddings = False):
        # x = self.base_model(input_ids = input_ids, attention_mask = attention_mask)
        x = F.dropout(inputs, p = self.dropout, training=self.training)
        x = self.linear(x)
        
        embeddings = x
        # # x = F.normalize(x, p=2, dim=-1)
        # # prottypes  = F.normalize(self.prottypes.weight, p = 2, dim = -1)
        # # x = x @ prottypes
        
        x = x@ self.prottypes.weight
        
        if return_embeddings:
            return x, embeddings
        return x



class NERModel(nn.Module):
    def __init__(self, encoder, hidden_dim = 256, prototype_embeddings_dim = 128, dropout = 0.25):
        super().__init__()
        self.base_model =  AutoModel.from_pretrained("emilyalsentzer/Bio_ClinicalBERT")
        self.ner_model = Classifier(self.base_model.embeddings.word_embeddings.embedding_dim, encoder.num_tags, hidden_dim=hidden_dim, prototype_embeddings_dim=prototype_embeddings_dim, dropout=dropout)
    def forward(self, input_ids, attention_mask, return_embeddings=False):
        outputs = self.base_model(input_ids = input_ids, attention_mask = attention_mask)
        return self.ner_model(outputs.last_hidden_state, return_embeddings = return_embeddings)

class EnsembleModel(nn.Module):
    def __init__(self, icd10_extration_model, disease_ner_model, prototype_embeddings_dim, hidden_dim,  num_classes, dropout = 0.25):
        super().__init__()
        self.icd10_extration_model = icd10_extration_model
        self.disease_ner_model = disease_ner_model
        for param in self.icd10_extration_model.parameters():
            param.requires_grad = False
        for param in self.disease_ner_model.parameters():
            param.requires_grad = False
            
        self.linear = nn.Sequential(
            nn.Linear(
                prototype_embeddings_dim * 2,
                hidden_dim
            ),
            nn.ReLU(),
            nn.Dropout(p = dropout),
            nn.Linear(hidden_dim, num_classes),
        )
    def forward(self, input_ids, attention_mask):
        
        icd10_outputs, icd10_embeddings = self.icd10_extration_model(input_ids = input_ids, attention_mask = attention_mask, return_embeddings=True)
        disease_ner_outputs, disease_ner_embeddings = self.disease_ner_model(input_ids = input_ids, attention_mask = attention_mask, return_embeddings=True)
        
        embeddings = torch.cat([icd10_embeddings, disease_ner_embeddings], dim=-1)
        outputs = self.linear(embeddings)
        
        return outputs

class MedicareCommercialModel(nn.Module):
    def __init__(self, model_num_labels,  hidden_dim = 256, prototype_embeddings_dim = 128, dropout = 0.25):
        super().__init__()
        self.base_model = AutoModel.from_pretrained("emilyalsentzer/Bio_ClinicalBERT")
        self.both_classifier = Classifier(self.base_model.embeddings.word_embeddings.embedding_dim, model_num_labels["both"], hidden_dim=hidden_dim, prototype_embeddings_dim=prototype_embeddings_dim, dropout=dropout)
        self.commercial_classifier = Classifier(self.base_model.embeddings.word_embeddings.embedding_dim, model_num_labels["commercial_only"], hidden_dim=hidden_dim, prototype_embeddings_dim=prototype_embeddings_dim, dropout=dropout)
        self.medicare_classifier = Classifier(self.base_model.embeddings.word_embeddings.embedding_dim, model_num_labels["medicare_only"], hidden_dim=hidden_dim, prototype_embeddings_dim=prototype_embeddings_dim, dropout=dropout)
        self.negation_classifier = Classifier(self.base_model.embeddings.word_embeddings.embedding_dim, 2, hidden_dim=hidden_dim, prototype_embeddings_dim=prototype_embeddings_dim, dropout=dropout)
    def forward(self, input_ids, attention_mask, all_tasks = False, task = None):
        embeddings =  self.base_model(input_ids = input_ids, attention_mask = attention_mask)
        if all_tasks:
            both_outputs = self.both_classifier(embeddings.last_hidden_state, return_embeddings = False)
            commercial_outputs = self.commercial_classifier(embeddings.last_hidden_state, return_embeddings = False)
            medicare_outputs = self.medicare_classifier(embeddings.last_hidden_state, return_embeddings = False)
            negation_outputs = self.negation_classifier(embeddings.last_hidden_state, return_embeddings = False)
            return {
                "both":both_outputs,
                "commercial_only":commercial_outputs,
                "medicare_only":medicare_outputs,
                "negation":negation_outputs
            }
        else:
            assert not task is None, "Either all_tasks should be True or task shoudn't be None"
            tasks = ["both", "commercial_only", "medicare_only", "negation"]
            assert task in tasks, "The task should be one of [{}]".format(",".join(tasks))
            if task == "both":
                current_model = self.both_classifier
            elif task == "commercial_only":
                current_model = self.commercial_classifier
            elif task == "medicare_only":
                current_model = self.medicare_classifier
            else:
                current_model = self.negation_classifier
            return current_model(embeddings.last_hidden_state, return_embeddings = False)
class FusedModel(nn.Module):
    def __init__(self, both_classifier, commercial_only_classifier, medicare_only_classifier, negation_classifier):
        super().__init__()
        self.both_classifier = both_classifier

        del self.both_classifier.commercial_classifier
        del self.both_classifier.medicare_classifier
        del self.both_classifier.negation_classifier


        self.commercial_only_classifier = commercial_only_classifier
        
        del self.commercial_only_classifier.both_classifier
        del self.commercial_only_classifier.medicare_classifier
        del self.commercial_only_classifier.negation_classifier
        

        self.medicare_only_classifier = medicare_only_classifier
        
        del self.medicare_only_classifier.both_classifier
        del self.medicare_only_classifier.commercial_classifier
        del self.medicare_only_classifier.negation_classifier

        self.negation_classifier = negation_classifier

        del self.negation_classifier.both_classifier
        del self.negation_classifier.commercial_classifier
        del self.negation_classifier.medicare_classifier
    def forward(self, input_ids, attention_mask):
        return {
            "medicare_commercial": self.both_classifier(input_ids = input_ids, attention_mask = attention_mask, task = "both"),
            "commercial_only": self.commercial_only_classifier(input_ids = input_ids, attention_mask = attention_mask, task = "commercial_only"),
            "medicare_only": self.medicare_only_classifier(input_ids = input_ids, attention_mask = attention_mask, task = "medicare_only"),
            "negation": self.negation_classifier(input_ids = input_ids, attention_mask = attention_mask, task = "negation"),
        }
class JointModel(nn.Module):
    def __init__(self, model_num_labels,  hidden_dim = 256, prototype_embeddings_dim = 128, dropout = 0.25):
        super().__init__()
        self.base_model = AutoModel.from_pretrained("emilyalsentzer/Bio_ClinicalBERT")
        self.both_classifier = Classifier(self.base_model.embeddings.word_embeddings.embedding_dim, model_num_labels["medicare_commercial"], hidden_dim=hidden_dim, prototype_embeddings_dim=prototype_embeddings_dim, dropout=dropout)
        self.commercial_classifier = Classifier(self.base_model.embeddings.word_embeddings.embedding_dim, model_num_labels["commercial_only"], hidden_dim=hidden_dim, prototype_embeddings_dim=prototype_embeddings_dim, dropout=dropout)
        self.medicare_classifier = Classifier(self.base_model.embeddings.word_embeddings.embedding_dim, model_num_labels["medicare_only"], hidden_dim=hidden_dim, prototype_embeddings_dim=prototype_embeddings_dim, dropout=dropout)

    def forward(self, input_ids, attention_mask):
        embeddings =  self.base_model(input_ids = input_ids, attention_mask = attention_mask)

        both_outputs = self.both_classifier(embeddings.last_hidden_state, return_embeddings = False)
        commercial_outputs = self.commercial_classifier(embeddings.last_hidden_state, return_embeddings = False)
        medicare_outputs = self.medicare_classifier(embeddings.last_hidden_state, return_embeddings = False)

        return {
            "medicare_commercial":both_outputs,
            "commercial_only":commercial_outputs,
            "medicare_only":medicare_outputs,
        }
       
class NegationModel(nn.Module):
    def __init__(self, base_model,  hidden_dim = 256, prototype_embeddings_dim = 128, dropout = 0.25):
        super().__init__()
        self.base_model = base_model
        
        self.negation_classifier = Classifier(self.base_model.embeddings.word_embeddings.embedding_dim, 2, hidden_dim=hidden_dim, prototype_embeddings_dim=prototype_embeddings_dim, dropout=dropout)

    def forward(self, input_ids, attention_mask):
        embeddings =  self.base_model(input_ids = input_ids, attention_mask = attention_mask)

        
        outputs = self.negation_classifier(embeddings.last_hidden_state, return_embeddings = False)

        return outputs
class SplitModel(nn.Module):
    def __init__(self, base_model, classifier):
        super().__init__()
        self.base_model = base_model 
        self.classifier = classifier 
    def forward(self, input_ids, attention_mask):
        embeddings =  self.base_model(input_ids = input_ids, attention_mask = attention_mask)

        
        outputs = self.classifier(embeddings.last_hidden_state, return_embeddings = False)

        return outputs
