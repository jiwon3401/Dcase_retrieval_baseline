import os
import pickle

import pandas as pd
from sentence_transformers import SentenceTransformer

import math
import torch
import torch.nn as nn
import numpy as np

from models.BERT_Config import MODELS


global_params = {
    "dataset_dir": "/home/user/dcase_retrieval/Clotho",
    "audio_splits": ["development", "validation", "evaluation"]
}

model_name = "sbert"
model = SentenceTransformer('all-mpnet-base-v2')  # 768-dimensional embeddings


# %%

text_embeds = {}

for split in global_params["audio_splits"]:

    text_fpath = os.path.join(global_params["dataset_dir"], f"{split}_text.csv")
    text_data = pd.read_csv(text_fpath)

    for i in text_data.index:
        tid = text_data.iloc[i].tid
        raw_text = text_data.iloc[i].raw_text

        print(split, tid, raw_text)

        text_embeds[tid] = model.encode(raw_text)
        
        

        
# Save text embeddings
embed_fpath = os.path.join(global_params["dataset_dir"], f"{model_name}_embeds.pkl")

with open(embed_fpath, "wb") as stream:
    pickle.dump(text_embeds, stream)

print("Save text embeddings to", embed_fpath)




#RoBERTa model embedding
'''
from transformers import RobertaTokenizer, RobertaModel
tokenizer = RobertaTokenizer.from_pretrained('roberta-base')

model_name = 'RoBERTa'

model = RobertaModel.from_pretrained('roberta-base')

text_embeds_roberta = {}

for split in global_params["audio_splits"]:

    text_fpath = os.path.join(global_params["dataset_dir"], f"{split}_text.csv")
    text_data = pd.read_csv(text_fpath)

    for i in text_data.index:
        tid = text_data.iloc[i].tid
        raw_text = text_data.iloc[i].raw_text

        print(split, tid, raw_text) 

        robert_embedding = model(**tokenizer(raw_text, return_tensors='pt'))
        text_embeds_roberta[tid]= robert_embedding[1].flatten().detach().numpy()
        
        
# Save text embeddings
embed_fpath = os.path.join(global_params["dataset_dir"], f"{model_name}_embeds.pkl")
with open(embed_fpath, "wb") as stream:
    pickle.dump(text_embeds_roberta, stream)
print("Save text embeddings to", embed_fpath)
        
'''