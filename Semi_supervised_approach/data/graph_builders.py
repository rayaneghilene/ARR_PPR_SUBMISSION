import numpy as np
import torch
from typing import List
from torch_geometric.data import Data, HeteroData
from sklearn.neighbors import NearestNeighbors


def build_knn_graph(embeddings: np.ndarray, k: int = 10, metric: str = 'cosine') -> Data:
    """Build an undirected kNN graph from embeddings and return a PyG Data object  """
    n = embeddings.shape[0]
    nbrs = NearestNeighbors(n_neighbors=k+1, metric=metric, n_jobs=-1).fit(embeddings)
    _, indices = nbrs.kneighbors(embeddings)

    edge_index = []
    for i in range(n):
        for j in indices[i, 1:]: 
            edge_index.append((i, int(j)))
            edge_index.append((int(j), i))  # undirected (both directions <-> ) 

    edge_index = np.asarray(edge_index, dtype=np.int64).T
    x = torch.from_numpy(embeddings)
    data = Data(x=x, edge_index=torch.from_numpy(edge_index))
    return data

