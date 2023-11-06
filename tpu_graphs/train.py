import torch
import random
from torch import nn, optim
from torch.utils.data import DataLoader

from .model import OneDModel
from .pt_loader import *

random.seed(42)
torch.manual_seed(0)

model = OneDModel(num_features=141)

criterion = nn.MSELoss()

optimizer = optim.SGD(model.parameters(), lr=0.01)

filenames = get_files('tile', 'valid')
dataset = LayoutDataset(filenames=filenames)
sampler = BufferedRandomSampler(len(dataset))
bs = 64
dataloader = DataLoader(dataset, batch_size=bs, shuffle=False, collate_fn=custom_collate_fn, sampler=sampler)

num_epochs = 10

for epoch in range(num_epochs):
    for data in dataloader:
        config_feat, node_feat, config_runtime, _, _ = data

        config_runtime = config_runtime.to(torch.float32)

        preds = model(node_feat, config_feat)
        loss = criterion(preds, config_runtime)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item()}')
