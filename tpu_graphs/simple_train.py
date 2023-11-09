from tqdm import tqdm
import wandb
import torch
import os
from torch import nn, optim
import numpy as np
from .model import SimpleModel, count_parameters

if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')


model = SimpleModel(hidden_channels=[128, 256, 512, 512, 1024], graph_feats=512)
count_parameters(model)

model.to(device)
criterion = nn.MSELoss()

optimizer = optim.Adam(model.parameters(), lr=3e-4)

TRAIN_DIR = 'data/npz_all/npz/tile/xla/valid/'
filenames = [os.path.join(TRAIN_DIR, filename) for filename in os.listdir(TRAIN_DIR)]
num_epochs = 5_000
bs = 4

wandb.init(project='tpu_graphs')

for epoch in tqdm(range(num_epochs)):
    for filename in filenames:
        graph_data = dict(np.load(filename, allow_pickle=True))
        node_feat = torch.from_numpy(graph_data['node_feat']).to(device)
        node_opcode = torch.from_numpy(graph_data['node_opcode']).to(device)
        edge_index = torch.from_numpy(graph_data['edge_index']).permute(1, 0).to(device)

        for trial_idx in range(0, len(graph_data['config_feat'][0]), bs):
            next_idx = min(trial_idx + bs, len(graph_data['config_feat'][0]))

            config_feat = torch.from_numpy(graph_data['config_feat'][trial_idx:next_idx]).to(device)
            config_runtime = torch.from_numpy(np.array([
                graph_data['config_runtime'][trial_idx:next_idx] / graph_data['config_runtime_normalizers'][trial_idx:next_idx]
            ])).flatten().to(torch.float32).to(device)

            config_runtime = config_runtime / 8.203627220003426
            node_feat = (node_feat - 14.231035232543945) / 305.2548828125
            config_feat = config_feat

            preds = model(config_feat, node_feat, node_opcode, edge_index).to(device)
            loss = torch.sqrt(criterion(preds, config_runtime))

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            wandb.log({'loss': loss.item()})

            if epoch % 1000 == 0:
                wandb.log({
                  'predictions': wandb.Histogram(preds.cpu().detach()),
                  'targets': wandb.Histogram(config_runtime.cpu().detach()),
                })

            break
        break

    wandb.log({'epoch': epoch})
