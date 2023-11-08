import wandb
import torch
import random
from torch import nn, optim
import numpy as np

from .model import SimpleModel, count_parameters
from .pt_loader import get_files

if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

print('running on', device)


random.seed(42)
torch.manual_seed(0)

model = SimpleModel(hidden_channels=[16, 32, 16, 48], graph_feats=64, hidden_dim=64)

count_parameters(model)

model.to(device)
criterion = nn.MSELoss()

optimizer = optim.Adam(model.parameters(), lr=3e-4)

filenames = get_files('tile', 'valid')
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


            if trial_idx % 2 == 0 and trial_idx != 0:
              break


    wandb.log({'epoch': epoch})
