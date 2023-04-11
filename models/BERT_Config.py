import os
import pickle

import pandas as pd
from sentence_transformers import SentenceTransformer

import math
import torch
import torch.nn as nn
import numpy as np

from transformers import BertModel, BertTokenizer, GPT2Model, GPT2Tokenizer,\
    RobertaModel, RobertaTokenizer, DistilBertModel, DistilBertTokenizer,\
    CLIPTokenizer, CLIPTextModel

MODELS = {
    'openai/clip-vit-base-patch32': (CLIPTextModel, CLIPTokenizer, 512),
    'prajjwal1/bert-tiny': (BertModel, BertTokenizer, 128),
    'prajjwal1/bert-mini': (BertModel, BertTokenizer, 256),
    'prajjwal1/bert-small': (BertModel, BertTokenizer, 512),
    'prajjwal1/bert-medium': (BertModel, BertTokenizer, 512),
    'gpt2': (GPT2Model, GPT2Tokenizer, 768),
    'distilgpt2': (GPT2Model, GPT2Tokenizer, 768),
    'bert-base-uncased': (BertModel, BertTokenizer, 768),
    'bert-large-uncased': (BertModel, BertTokenizer, 1024),
    'roberta-base': (RobertaModel, RobertaTokenizer, 768),
    'roberta-large': (RobertaModel, RobertaTokenizer, 1024),
    'distilbert-base-uncased': (DistilBertModel, DistilBertTokenizer, 768),
    "distilroberta-base": (RobertaModel, RobertaTokenizer, 768),
}



global_params = {
    "dataset_dir": "/home/user/dcase_retrieval/Clotho",
    "audio_splits": ["development", "validation", "evaluation"]
}

