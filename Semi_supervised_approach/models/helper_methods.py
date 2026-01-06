from sklearn.metrics import roc_auc_score
from typing import List, Optional
import torch
import numpy as np

def labels_to_tensor(labels: List[int], num_classes: Optional[int] = None) -> torch.Tensor:
    y = torch.tensor(labels, dtype=torch.long)
    if num_classes is not None:
        assert y.max().item() < num_classes
    return y

def sample_negative_edges(num_nodes: int, num_samples: int, forbid=set()) -> np.ndarray:
    # uniform sampling of negatives
    negs = set()
    while len(negs) < num_samples:
        u = np.random.randint(0, num_nodes)
        v = np.random.randint(0, num_nodes)
        if u == v: continue
        if (u, v) in forbid: continue
        negs.add((u, v))
    arr = np.array(list(negs), dtype=np.int64).T
    return arr


def compute_auc(y_true, y_score):
    try:
        return roc_auc_score(y_true, y_score)
    except Exception:
        return 0.5