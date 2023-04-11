import dbm
import json
import os
import shelve
from dbm import dumb

import nltk
import torch

from ast import literal_eval
import numpy as np
import pandas as pd
import time

from utils import criterion_utils, data_utils, model_utils
#import criterion_utils, data_utils, model_utils

start1 = time.time()


# Trial info
trial_base = '/home/user/dcase_retrieval/outputs' 
trial_series = 'audio_text_retrieval' 
trial_name = '2023-04-11_bb24b_00000' 
ckp_dir = 'checkpoint_13' 


dbm._defaultmod = dumb
dbm._modules = {"dbm.dumb": dumb}

stopwords = nltk.corpus.stopwords.words("english")


# Model checkpoint directory
ckp_fpath = os.path.join(trial_base, trial_series, trial_name, ckp_dir)

# Load trial parameters
conf_fpath = os.path.join(trial_base, trial_series, trial_name, "params.json")
with open(conf_fpath, "rb") as store:
    conf = json.load(store)
print("Load", conf_fpath)

# Load data
data_conf = conf["data_conf"]
train_ds = data_utils.load_data(data_conf["train_data"])
val_ds = data_utils.load_data(data_conf["val_data"])
eval_ds = data_utils.load_data(data_conf["eval_data"])

# Restore model checkpoint
param_conf = conf["param_conf"]
model_params = conf[param_conf["model"]]
obj_params = conf["criteria"][param_conf["criterion"]]
model = model_utils.init_model(model_params, train_ds.text_vocab)
model = model_utils.restore(model, ckp_fpath)
print(model)



model.eval()

for name, ds in zip(["train", "val", "eval"], [train_ds, val_ds, eval_ds]):
    text2vec = {}
    for idx in ds.text_data.index:
        item = ds.text_data.iloc[idx]

        if ds.text_level == "word":
            text_vec = torch.as_tensor([ds.text_vocab(key) for key in item["tokens"] if key not in stopwords])
            text2vec[item["tid"]] = torch.unsqueeze(text_vec, dim=0)

        elif ds.text_level == "sentence":
            text_vec = torch.as_tensor([ds.text_vocab(item["tid"])])
            text2vec[item["tid"]] = torch.unsqueeze(text_vec, dim=0)

    # Compute pairwise cross-modal scores
    score_fpath = os.path.join(ckp_fpath, f"{name}_xmodal_scores.db")
    with shelve.open(filename=score_fpath, flag="n", protocol=2) as stream:
        for fid in ds.text_data["fid"].unique():
            group_scores = {}

            # Encode audio data
            audio_vec = torch.as_tensor(ds.audio_data[fid][()])
            audio_vec = torch.unsqueeze(audio_vec, dim=0)
            audio_embed = model.audio_branch(audio_vec)[0]

            for tid in text2vec:
                text_embed = model.text_branch(text2vec[tid])[0]  # Encode text data
                xmodal_S = criterion_utils.score(audio_embed, text_embed, obj_params["args"].get("dist", "cosine_similarity"))
                group_scores[tid] = xmodal_S.item()
            stream[fid] = group_scores
    print("Save", score_fpath)

    
end1 = time.time()
print("retrieval_score_time: ", end1 - start1)



####retrieval_csv_file (xmodal_retrieval.py)
data_conf = conf["data_conf"]


def measure(qid2items):
    # Retrieval metrics over sample queries
    # i.e., recall@{1, 5, 10}, mAP@10.
    qid_R1s, qid_R5s, qid_R10s, qid_mAP10s = [], [], [], []

    for qid in qid2items:
        items = qid2items[qid]

        objects = [i[0] for i in items]
        scores = np.array([i[1] for i in items])
        targets = np.array([i[2] for i in items])

        desc_indices = np.argsort(scores, axis=-1)[::-1]
        targets = np.take_along_axis(arr=targets, indices=desc_indices, axis=-1)

        # Recall at cutoff K
        R1 = np.sum(targets[:1], dtype=float) / np.sum(targets, dtype=float)
        R5 = np.sum(targets[:5], dtype=float) / np.sum(targets, dtype=float)
        R10 = np.sum(targets[:10], dtype=float) / np.sum(targets, dtype=float)

        qid_R1s.append(R1)
        qid_R5s.append(R5)
        qid_R10s.append(R10)

        # Mean average precision
        # positions = np.arange(1, len(targets) + 1, dtype=float)[targets > 0]
        positions = np.arange(1, 11, dtype=float)[targets[:10] > 0]
        if len(positions) > 0:
            precisions = np.divide(np.arange(1, len(positions) + 1, dtype=float), positions)
            avg_precision = np.sum(precisions, dtype=float) / np.sum(targets, dtype=float)
            qid_mAP10s.append(avg_precision)
        else:
            qid_mAP10s.append(0.0)

    print("mAP: {:.3f}".format(np.mean(qid_mAP10s)),
          "R1: {:.3f}".format(np.mean(qid_R1s)),
          "R5: {:.3f}".format(np.mean(qid_R5s)),
          "R10: {:.3f}".format(np.mean(qid_R10s)), end="\n")


for name in data_conf:
    params = data_conf[name]
    name = name.replace("_data", "")

    # Load text data
    text_fpath = os.path.join(params["dataset"], params["text_data"])
    text_data = pd.read_csv(text_fpath, converters={"tokens": literal_eval})
    print("Load", text_fpath, end="\n\n")

    # Process text data
    tid2fid, tid2text, fid2fname = {}, {}, {}
    for idx in text_data.index:
        item = text_data.iloc[idx]
        tid2fid[item["tid"]] = item["fid"]
        tid2text[item["tid"]] = item["text"]
        fid2fname[item["fid"]] = item["fname"]

    # Load cross-modal scores
    score_fpath = os.path.join(ckp_fpath, f"{name}_xmodal_scores.db")
    with shelve.open(filename=score_fpath, flag="r", protocol=2) as stream:
        fid2items, tid2items = {}, {}
        for fid in stream:
            group_scores = stream[fid]

            # Audio2Text retrieval
            fid2items[fid] = [(tid, group_scores[tid], tid2fid[tid] == fid) for tid in group_scores]

            # Text2Audio retrieval
            for tid in group_scores:
                if tid2items.get(tid, None) is None:
                    tid2items[tid] = [(fid, group_scores[tid], tid2fid[tid] == fid)]
                else:
                    tid2items[tid].append((fid, group_scores[tid], tid2fid[tid] == fid))

        print("Audio2Text retrieval")
        measure(fid2items)
        print("Text2Audio retrieval")
        measure(tid2items)

        # Output Text2Audio retrieval results
        results = []
        for tid in tid2items:
            items = tid2items[tid]
            objects = [i[0] for i in items]
            scores = np.array([i[1] for i in items])
            targets = np.array([i[2] for i in items])
            desc_indices = np.argsort(scores, axis=-1)[::-1]
            line = [tid2text[tid]] + [fid2fname[objects[idx]] for idx in desc_indices[:10]]
            results.append(line)

        results = pd.DataFrame(data=results)
        result_fpath = os.path.join(ckp_fpath, f"{name}.t2a_retrieval.csv")
        results.to_csv(result_fpath, index=False)
        print("Save", result_fpath, end="\n\n")
        print("===================================")

