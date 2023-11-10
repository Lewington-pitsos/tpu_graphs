from tqdm import tqdm
import wandb
import torch
import os
from torch import nn, optim
import numpy as np
from .model import SimpleModel, count_parameters, ConfigDense
from .plot import *
from .conly import file_preds
from .pt_loader import file_data, FilewiseLoader

if torch.cuda.is_available():
	device = torch.device('cuda')
else:
	device = torch.device('cpu')

def valid_loss(model, criterion):
	VALID_DIR = 'data/npz_all/npz/tile/xla/valid/'
	valid_filenames = [os.path.join(VALID_DIR, filename) for filename in os.listdir(VALID_DIR)]

	losses = []
	for filename in valid_filenames:
		preds, config_runtime = file_preds(filename, model, 512, device)
		preds, config_runtime = torch.from_numpy(preds), torch.from_numpy(config_runtime)
		loss = torch.sqrt(criterion(preds, config_runtime))
		losses.append(loss.mean().item())

	return np.mean(losses)

# model = ConfigDense(in_channels=24, out_channels=512, hidden=512)
model = SimpleModel(hidden_channels=[128, 256, 512, 512, 1024], graph_feats=512)
num_parametrs = count_parameters(model)
model.to(device)

criterion = nn.MSELoss()
TRAIN_DIR = 'data/npz_all/npz/tile/xla/train/'
filenames = [os.path.join(TRAIN_DIR, filename) for filename in os.listdir(TRAIN_DIR)][:10]

config = {
	'num_epochs': 4,
	'batch_size': 1,
	'n_files': len(filenames),
	'lr': 3e-4,
	'num_parametrs': num_parametrs,
	'model_class': type(model).__name__
}


# loader = FilewiseLoader(filenames, device, 64)
optimizer = optim.Adam(model.parameters(), lr=config['lr'])
wandb.init(project='tpu_graphs', config=config)

itr = 0
log_every = 5_000
next_log = 0
for epoch in range(config['num_epochs']):
	wandb.log({'valid_loss': valid_loss(model, criterion)})

	for filename in tqdm(filenames):
		node_feat, node_opcode, edge_index, config_feat, config_runtime = file_data(filename, device)

		for trial_idx in range(0, len(config_runtime), config['batch_size']):
			itr += config['batch_size']
			next_idx = trial_idx + config['batch_size']

			batch_config = config_feat[trial_idx:next_idx]
			batch_runtime = config_runtime[trial_idx:next_idx]

			preds = model(batch_config, node_feat, node_opcode, edge_index)
			loss = torch.sqrt(criterion(preds.flatten(), batch_runtime))

			loss.backward()
			optimizer.step()
			optimizer.zero_grad()

			wandb.log({'loss': loss.item()})

		if itr >= next_log:
			plot_outputs_vs_predictions(batch_runtime.cpu().detach(), preds.cpu().detach())
			plot_config(batch_config.cpu().detach())
			next_log += log_every

	wandb.log({'epoch': epoch})


wandb.log({'valid_loss': valid_loss(model, criterion)})

torch.save(model.state_dict(), 'model.pt')
