import wandb
import torch
from torch import Tensor
from torch_geometric.nn import GCNConv, global_mean_pool
from torch_geometric.datasets import Planetoid
import torch.nn.functional as F
from .model import GCN

n_feats = 128
n_nodes = 37
n_edges = 350

nodes = torch.randn(n_nodes, n_feats)
edges = torch.randint(0, n_nodes, (2, n_edges))
batch_map = torch.tensor([0] * 21 + [1] * 9 + [2] * 7)

print(edges.shape)

model = GCN(n_feats, 16)

out = model(nodes, edges, batch_map)

print(out.shape)
print(out)


# optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

# wandb.init(project='planet_gnn')

# model.to(device)

# data = data.to(device)


# for epoch in range(200):


#     pred = model(data.x, data.edge_index)
#     loss = F.cross_entropy(pred[data.train_mask], data.y[data.train_mask])

#     optimizer.zero_grad()
#     loss.backward()
#     optimizer.step()

#     wandb.log({'loss': loss.item()})
