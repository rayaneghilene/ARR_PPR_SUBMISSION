import torch
import torch.nn as nn

class DotProductDecoder(nn.Module):
    def forward(self, z, edge_index_pos, edge_index_neg=None):
        # z: [N, D]; edge_index: [2, E]
        src = edge_index_pos[0]
        dst = edge_index_pos[1]
        scores_pos = (z[src] * z[dst]).sum(dim=-1)
        scores = {'pos': scores_pos}
        if edge_index_neg is not None:
            srcn = edge_index_neg[0]
            dstn = edge_index_neg[1]
            scores_neg = (z[srcn] * z[dstn]).sum(dim=-1)
            scores['neg'] = scores_neg
        return scores


class MLPDecoder(nn.Module):
    def __init__(self, dim, hidden=128):
        super().__init__()
        self.mlp = nn.Sequential(nn.Linear(dim*2, hidden), nn.ReLU(), nn.Linear(hidden, 1))
    def forward(self, z, edge_index_pos, edge_index_neg=None):
        def score_edges(ei):
            src = ei[0]
            dst = ei[1]
            h = torch.cat([z[src], z[dst]], dim=-1)
            return self.mlp(h).squeeze(-1)
        scores = {'pos': score_edges(edge_index_pos)}
        if edge_index_neg is not None:
            scores['neg'] = score_edges(edge_index_neg)
        return scores
