import torch 
from torch import nn 
from torch.nn import functional as F
from transformers import AutoModel
from tqdm.auto import tqdm
import pickle


def load_medicare_commercial_model(path):
    return torch.load(path, map_location=torch.device('cpu'))