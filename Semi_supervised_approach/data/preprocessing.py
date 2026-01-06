import random
import torch
import torch.nn as nn
from torch_geometric.data import Data
import pandas as pd
from typing import List, Dict, Optional
import numpy as np
import networkx as nx
from sklearn.metrics.pairwise import cosine_similarity

def load_clean_dataset(path: str) -> pd.DataFrame:
    """Load and clean the French dataset.."""
    df = pd.read_excel(path)
    df = df.drop(columns=[f"Unnamed: {i}" for i in range(2, 33)], errors="ignore")
    df = df.drop(columns=["Unnamed: 0"], errors="ignore")
    df.columns = df.iloc[3]
    df = df.iloc[4:].reset_index(drop=True)
    return df

def add_attribute_co_membership_edges(data, attributes, max_edges_per_node=5, embeddings=None):
    """
    This is the method defined in section 4.2.1 of the paper. 
    For each attribute bucket, add up to max_edges_per_node edges per node.
    If embeddings provided, prefer nearest neighbors within the bucket.
    """
    n = data.x.shape[0]
    # print(n)

    existing = set((int(u), int(v)) for u, v in data.edge_index.cpu().numpy().T)
    for attr_vals in attributes.values():
        # bucket nodes by value
        buckets = {}
        for i, v in enumerate(attr_vals):
            buckets.setdefault(v, []).append(i)

        for members in buckets.values():
            m = len(members)
            if m <= 1:
                continue
            if embeddings is not None and m > max_edges_per_node:
                # compute distances inside bucket
                mem_arr = np.array(members)
                emb_sub = embeddings[mem_arr]  # shape [m, d]
                # simple pairwise cosine distances
                sim = cosine_similarity(emb_sub)
                for idx_i, node in enumerate(mem_arr):
                    # exclude self
                    sims = sim[idx_i].copy()
                    sims[idx_i] = -1
                    topk = sims.argsort()[-max_edges_per_node:]
                    for tidx in topk:
                        nbr = int(mem_arr[tidx])
                        existing.add((node, nbr))
                        existing.add((nbr, node))
            else:
                # small bucket or no embeddings: random sampling.. We can workshop this later.
                for node in members:
                    # sample up to max_edges_per_node other members
                    choices = [m for m in members if m != node]
                    random.shuffle(choices)
                    for nbr in choices[:max_edges_per_node]:
                        existing.add((node, nbr))
                        existing.add((nbr, node))

    edge_index = np.array(list(existing), dtype=np.int64).T
    data.edge_index = torch.from_numpy(edge_index)
    return data
