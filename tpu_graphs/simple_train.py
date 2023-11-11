from tqdm import tqdm
import wandb
import torch
import os
from torch import nn, optim
import numpy as np
from .model import SimpleModel, count_parameters, ConfigDense
from .plot import *
from .validate import valid_loss
from .pt_loader import file_data, FilewiseLoader

if torch.cuda.is_available():
	device = torch.device('cuda')
else:
	device = torch.device('cpu')

# model = ConfigDense(in_channels=24, out_channels=512, hidden=512)
model = SimpleModel(hidden_channels=[128, 256, 512, 512, 1024], graph_feats=512)
num_parametrs = count_parameters(model)
model.to(device)

criterion = lambda x, y: torch.sqrt(nn.MSELoss()(x, y))
TRAIN_DIR = 'data/npz_all/npz/tile/xla/train/'
filenames = [os.path.join(TRAIN_DIR, filename) for filename in os.listdir(TRAIN_DIR)]

config = {
	'num_epochs': 2000, # 2
	'batch_size': 4, # 256
	'n_files': len(filenames),
	'lr': 3e-4,
	'num_parametrs': num_parametrs,
	'model_class': type(model).__name__
}

loader = FilewiseLoader(filenames, device, config['batch_size'])
optimizer = optim.Adam([
	{ 'params': model.conv_parameters, 'lr': config['lr'] / 4 },
	{ 'params': model.dense_parameters, 'lr': config['lr']},
])
wandb.init(project='overfit_tpu_graphs', config=config)

itr = 0
for epoch in range(config['num_epochs']):

	for node_feat, node_opcode, edge_index, batch_config, batch_runtime in tqdm(loader):
		itr += 1
		preds = model(batch_config, node_feat, node_opcode, edge_index)
		loss = criterion(preds.flatten(), batch_runtime)

		loss.backward()
		optimizer.step()
		optimizer.zero_grad()

		wandb.log({'loss': loss.item()})

		if itr % 1000 == 0:
			plot_outputs_vs_predictions(batch_runtime.cpu().detach(), preds.cpu().detach())
			plot_config(batch_config.cpu().detach())

		if itr % 2 == 0:
			loader.reset()
			break

	# wandb.log({f'valid_{key}': val for key, val in valid_loss(model, criterion, device).items()})
	wandb.log({'epoch': epoch})

wandb.finish()

torch.save(model.state_dict(), 'model.pt')
