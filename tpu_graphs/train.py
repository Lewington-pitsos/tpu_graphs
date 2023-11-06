import wandb
import torch
import random
from torch import nn, optim
from torch.utils.data import DataLoader

from .model import OneDModel
from .pt_loader import *

if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

print('running on', device)


random.seed(42)
torch.manual_seed(0)

model = OneDModel(num_features=141)
model.to(device)
criterion = nn.MSELoss()

optimizer = optim.Adam(model.parameters(), lr=3e-4)

filenames = get_files('tile', 'valid')
dataset = LayoutDataset(filenames=filenames[:1])
# sampler = BufferedRandomSampler(len(dataset), buffer_size=4000)
bs = 128
dataloader = DataLoader(dataset, batch_size=bs, shuffle=True, collate_fn=custom_collate_fn)

num_epochs = 100

wandb.init(project='tpu_graphs')

for epoch in range(num_epochs):
    for i, data in enumerate(dataloader):
        config_feat, node_feat, config_runtime, _, _ = data

        config_feat = config_feat.to(device)
        node_feat = node_feat.to(device)
        config_runtime = config_runtime.to(device)

        config_runtime = config_runtime.to(torch.float32)

        preds = model(node_feat, config_feat)
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
