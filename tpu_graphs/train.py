import wandb
import torch
import random
from torch import nn, optim
from torch.utils.data import DataLoader

from .model import GCN
from .pt_loader import *

if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

print('running on', device)


random.seed(42)
torch.manual_seed(0)

model = GCN(165, 64)
model.to(device)
criterion = nn.MSELoss()

optimizer = optim.Adam(model.parameters(), lr=3e-4)

filenames = get_files('tile', 'valid')
dataset = LayoutDataset(filenames=filenames[:1])
num_epochs = 10

wandb.init(project='tpu_graphs')

for epoch in range(num_epochs):
    for i, data in enumerate(dataset):
        config_feat, node_feat, config_runtime, edge_index, _, _ = data

        config_feat = torch.from_numpy(config_feat).to(device)
        node_feat = torch.from_numpy(node_feat).to(device)
        config_runtime = torch.from_numpy(config_runtime).to(torch.float32).to(device)
        edge_index = torch.from_numpy(edge_index).permute(1, 0).to(device)

        node_feat = torch.concat([node_feat, config_feat.repeat(node_feat.shape[0], 1)], dim=1).to(device)

        preds = model(node_feat, edge_index, torch.zeros(node_feat.shape[0], dtype=torch.long).to(device))
        loss = torch.sqrt(criterion(preds, config_runtime))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if i % 1500 == 0:

            p = preds.detach().to('cpu')
            wandb.log({
                'config_feat': wandb.Histogram(config_feat.to('cpu')),
                'config_feat_mean': config_feat.to('cpu').mean(),
                'config_feat_std': config_feat.to('cpu').std(),
                'node_feat': wandb.Histogram(node_feat.to('cpu')),
                'node_feat_mean': node_feat.to('cpu').mean(),
                'node_feat_std': node_feat.to('cpu').std(),
                'config_runtime': wandb.Histogram(config_runtime.to('cpu')),
                'config_runtime_mean': config_runtime.to('cpu').mean(),
                'config_runtime_std': config_runtime.to('cpu').std(),
                'preds': wandb.Histogram(p),
                'preds_mean': p.mean(),
                'preds_std': p.std(),
            })


        wandb.log({'loss': loss.item()})
    wandb.log({'epoch': epoch})
