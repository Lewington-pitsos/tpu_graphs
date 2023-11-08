import wandb
import torch
from torch import Tensor
from torch_geometric.nn import GCNConv, global_mean_pool
from torch_geometric.datasets import Planetoid
import torch.nn.functional as F
from .model import SimpleModel, count_parameters

model = SimpleModel(hidden_channels=[16, 32, 16, 48], graph_feats=64, hidden_dim=64)
count_parameters(model)

# n_feats = 128
# n_nodes = 37
# n_edges = 350

# nodes = torch.randn(n_nodes, n_feats)
# edges = torch.randint(0, n_nodes, (2, n_edges))
# batch_map = torch.tensor([0] * 21 + [1] * 9 + [2] * 7)

# print(edges.shape)

# model = GCN(n_feats, 16)

# out = model(nodes, edges, batch_map)

# print(out.shape)
# print(out)
