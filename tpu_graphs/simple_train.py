import wandb
import torch
import os
from torch import nn, optim
import numpy as np
from torch_geometric.nn import GCNConv, global_mean_pool
from torch import Tensor

from .model import SimpleModel

if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')


class SimpleModel(torch.nn.Module):
    def __init__(self, hidden_channels, graph_feats, hidden_dim):
        super().__init__()

        op_embedding_dim = 4  # I choose 4-dimensional embedding
        self.embedding = torch.nn.Embedding(120,  # 120 different op-codes
                                            op_embedding_dim,
                                           )
        assert len(hidden_channels) > 0
        in_channels = op_embedding_dim + 140
        self.convs = torch.nn.ModuleList()
        last_dim = hidden_channels[0]

        # Create a sequence of Graph Convolutional Network (GCN) layers
        self.convs.append(GCNConv(in_channels, hidden_channels[0]))
        for i in range(len(hidden_channels) - 1):
            self.convs.append(GCNConv(hidden_channels[i], hidden_channels[i+1]))
            last_dim = hidden_channels[i+1]
        self.convs.append(GCNConv(last_dim, graph_feats))

        # Define a sequential dense neural network
        self.dense = torch.nn.Sequential(nn.Linear(graph_feats + 24, 64),
                                         nn.ReLU(),
                                         nn.Linear(64, 64),
                                         nn.ReLU(),
                                         nn.Linear(64, 1),
                                        )

    def forward(self, x_cfg: Tensor, x_feat: Tensor, x_op: Tensor, edge_index: Tensor) -> Tensor:
        x = torch.cat([x_feat, self.embedding(x_op.long())], dim=1)
        for conv in self.convs:
            x = conv(x, edge_index).relu()

        x_graph = torch.mean(x, 0)

        x = torch.cat([x_cfg, x_graph.repeat((len(x_cfg), 1))], axis=1)
        x = torch.flatten(self.dense(x))

        return x

model = SimpleModel(hidden_channels=[16, 32, 16, 48], graph_feats=64, hidden_dim=64)

model.to(device)
criterion = nn.MSELoss()

optimizer = optim.Adam(model.parameters(), lr=3e-4)

TRAIN_DIR = 'data/npz_all/npz/tile/xla/valid/'
filenames = [os.path.join(TRAIN_DIR, filename) for filename in os.listdir(TRAIN_DIR)]
num_epochs = 5_000

wandb.init(project='tpu_graphs')

for epoch in range(num_epochs):
    for filename in filenames[1:2]:
        graph_data = dict(np.load(filename, allow_pickle=True))

        for trial_idx in range(len(graph_data['config_feat'][0])):
            config_feat = torch.from_numpy(graph_data['config_feat'][trial_idx]).to(device)
            node_feat = torch.from_numpy(graph_data['node_feat']).to(device)
            node_opcode = torch.from_numpy(graph_data['node_opcode']).to(device)
            config_runtime = torch.from_numpy(np.array([
                graph_data['config_runtime'][trial_idx] / graph_data['config_runtime_normalizers'][trial_idx]
            ])).to(torch.float32).to(device)
            edge_index = torch.from_numpy(graph_data['edge_index']).permute(1, 0).to(device)

            config_runtime = config_runtime / 8.203627220003426
            node_feat = (node_feat - 14.231035232543945) / 305.2548828125
            config_feat = config_feat.unsqueeze(0)

            preds = model(config_feat, node_feat, node_opcode, edge_index).to(device)
            loss = torch.sqrt(criterion(preds, config_runtime))

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            wandb.log({'loss': loss.item()})

            if epoch % 1000 == 0:
                print('pred', preds.item(), 'actual', config_runtime.item())


            if trial_idx % 32 == 0 and trial_idx != 0:
              break

    wandb.log({'epoch': epoch})
