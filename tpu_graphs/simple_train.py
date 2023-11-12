from tqdm import tqdm
import wandb
import torch
import os
from torch import nn, optim
import numpy as np
from .model import GraphModel, count_parameters, ConfigDense, Opcodes
from .plot import *
from .validate import valid_loss
from .pt_loader import file_data, FilewiseLoader

if torch.cuda.is_available():
	device = torch.device('cuda')
else:
	device = torch.device('cpu')


# model = ConfigDense(in_channels=24, out_channels=512, hidden=512)
model = Opcodes(in_channels=24, out_channels=128, hidden=128, op_embedding_dim=128)
# model = GraphModel(hidden_channels=[128, 256, 512, 512, 1024], graph_feats=512)
num_parametrs = count_parameters(model)
model.to(device)

criterion = lambda x, y: torch.sqrt(nn.MSELoss()(x, y))
TRAIN_DIR = 'data/npz_all/npz/tile/xla/train/'
filenames = [os.path.join(TRAIN_DIR, filename) for filename in os.listdir(TRAIN_DIR)]

config = {
	'num_epochs': 10,
	'batch_size': 256,
	'n_files': len(filenames),
	'lr': 5e-4,
	'num_parametrs': num_parametrs,
	'model_class': type(model).__name__
}

loader = FilewiseLoader(filenames, device, config['batch_size'])
optimizer = optim.Adam(model.parameters(), lr=config['lr'])
wandb.init(project='tpu_graphs', config=config)

for epoch in range(config['num_epochs']):
	plotted = False
	for node_feat, node_opcode, edge_index, batch_config, batch_runtime in tqdm(loader):
		# node_feat = (n_nodes, n_feat)
		# node_opcode =  (n_nodes)
		# edge_index = (2, n_edges)
		# batch_config = (batch_size, 24)
		# batch_runtime = (batch_size)

		preds = model(batch_config, node_feat, node_opcode, edge_index)
		loss = criterion(preds.flatten(), batch_runtime)

		loss.backward()
		optimizer.step()
		optimizer.zero_grad()

		wandb.log({'loss': loss.item()})

		if not plotted:
			plotted = True
			plot_outputs_vs_predictions(batch_runtime.cpu().detach(), preds.cpu().detach())
			plot_config(batch_config.cpu().detach(), 'tile_config')
			plot_config(node_feat.cpu().detach(), 'node_feat_plot')
			plot_opcodes(node_opcode.cpu().detach())

	wandb.log({f'valid_{key}': val for key, val in valid_loss(model, criterion, device).items()})
	wandb.log({'epoch': epoch})

wandb.finish()

torch.save(model.state_dict(), 'model.pt')
