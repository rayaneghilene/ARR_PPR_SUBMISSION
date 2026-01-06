import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, SAGEConv, GATConv, JumpingKnowledge
from torch_geometric.nn.norm import BatchNorm, GraphNorm

class NodeEncoder(nn.Module):
    """
    Multi-layer GNN encoder with:
      - per-layer: Conv -> Norm -> Activation -> Dropout
      - residual connections 
      - JumpingKnowledge (lstm / concat / max)
      - final projection head (for downstream eval)
    Works for 'gcn', 'graphsage', 'gat' :)
    """

    def __init__(
        self,
        encoder: str,
        in_dim: int,
        hidden_dim: int = 128,
        out_dim: int = 128,
        num_layers: int = 3,
        dropout: float = 0.5,
        use_input_mlp: bool = True,
        activation: str = "relu",   # "relu" or "elu"
        norm: str = "batch",        # "batch", "graph", or None
        residual: bool = True,
        jk_mode: str = None,        # None, "cat", "max", "lstm", "last"
        gat_heads: int = 4,
        gat_concat: bool = True,
        use_proj: bool = True,
    ):
        super().__init__()
        enc = encoder.lower()
        assert enc in {"gcn", "graphsage", "gat"}, f"Unsupported encoder {encoder}"
        self.encoder = enc
        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
        self.out_dim = out_dim
        self.num_layers = num_layers
        self.dropout = dropout
        self.residual = residual
        self.jk_mode = jk_mode
        self.use_proj = use_proj
        self.gat_heads = gat_heads
        self.gat_concat = gat_concat

        # Input MLP to transform raw features -> hidden_dim
        if use_input_mlp:
            self.input_mlp = nn.Sequential(
                nn.Linear(in_dim, hidden_dim),
                nn.ReLU() if activation == "relu" else nn.ELU(),
                nn.Dropout(dropout),
            )
            cur_in = hidden_dim
        else:
            self.input_mlp = None
            cur_in = in_dim

        # Build layers list
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        self.res_proj = nn.ModuleList() if residual else None

        for i in range(num_layers):
            in_ch = cur_in if i == 0 else hidden_dim
            # For GAT handle per-head sizing so the conv outputs hidden_dim consistently (4 yielded the best results so far)
            if self.encoder == "gcn":
                conv = GCNConv(in_ch, hidden_dim)
            elif self.encoder == "graphsage":
                conv = SAGEConv(in_ch, hidden_dim)
            elif self.encoder == "gat":
                # pick out_channels per head so total out == hidden_dim when concat=True
                if gat_concat:
                    assert hidden_dim % gat_heads == 0, "hidden_dim must be divisible by heads when concat=True"
                    out_per_head = hidden_dim // gat_heads
                    conv = GATConv(in_ch, out_per_head, heads=gat_heads, concat=True)
                else:
                    # concat=False: each head output is averaged, so specify out channels directly
                    conv = GATConv(in_ch, hidden_dim, heads=gat_heads, concat=False)
            self.convs.append(conv)

            # Normalization layer
            if norm == "batch":
                self.norms.append(BatchNorm(hidden_dim))
            elif norm == "graph":
                self.norms.append(GraphNorm(hidden_dim))
            else:
                self.norms.append(nn.Identity())

            # Residual projection if dims differ
            if residual:
                # residual adds input of this layer to output of conv
                # need to project input (in_ch) -> hidden_dim if different
                if in_ch != hidden_dim:
                    self.res_proj.append(nn.Linear(in_ch, hidden_dim))
                else:
                    self.res_proj.append(nn.Identity())

        # JumpingKnowledge
        if jk_mode is not None:
            # 'cat' (aka concatenate) will produce hidden_dim * num_layers 
            self.jk = JumpingKnowledge(jk_mode, channels=hidden_dim, num_layers=num_layers)
            final_in = hidden_dim * num_layers if jk_mode == "cat" else hidden_dim
        else:
            self.jk = None
            final_in = hidden_dim

        # Final projection head
        if use_proj:
            self.proj = nn.Sequential(
                nn.Linear(final_in, out_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
            )
        else:
            self.proj = nn.Identity()

        # Activation choice (elu performed best for graphsage during the hyperparam optimization)
        self.act = (lambda x: F.relu(x)) if activation == "relu" else (lambda x: F.elu(x))

    def forward(self, x, edge_index):
        # x: node features [N, in_dim]
        h = x
        if self.input_mlp is not None:
            h = self.input_mlp(h)  # -> hidden_dim

        layer_outputs = []
        for i, conv in enumerate(self.convs):
            h_in = h  # for residual
            # conv
            if self.encoder == "gat":
                # GATConv expects (x, edge_index) and handles heads internally
                h = conv(h, edge_index)
            else:
                h = conv(h, edge_index)
            # norm, activation, dropout
            h = self.norms[i](h)
            h = self.act(h)
            h = F.dropout(h, p=self.dropout, training=self.training)

            # residual
            if self.residual:
                res = self.res_proj[i](h_in)
                h = h + res

            layer_outputs.append(h)

        if self.jk is not None:
            h = self.jk(layer_outputs)
        else:
            h = layer_outputs[-1]

        h = self.proj(h)  # project to out_dim if use_proj
        return h