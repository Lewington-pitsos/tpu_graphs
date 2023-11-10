from tqdm import tqdm
import wandb
import torch
import os
from torch import nn, optim
import numpy as np
from .model import SimpleModel, count_parameters, ConfigDense
from .plot import *
from .conly import file_preds

if torch.cuda.is_available():
	device = torch.device('cuda')
else:
	device = torch.device('cpu')

def valid_loss(model, bs, criterion):
	VALID_DIR = 'data/npz_all/npz/tile/xla/valid/'
	valid_filenames = [os.path.join(VALID_DIR, filename) for filename in os.listdir(VALID_DIR)]

	losses = []
	for filename in valid_filenames:
		preds = file_preds(filename, model, bs, device)
		loss = torch.sqrt(criterion(preds, config_runtime))
		losses.append(loss.mean().item())

	return np.mean(losses)

model = ConfigDense(in_channels=24, out_channels=512, hidden=512)
# model = SimpleModel(hidden_channels=[128, 256, 512, 512, 1024], graph_feats=512)
count_parameters(model)

model.to(device)
criterion = nn.MSELoss()

optimizer = optim.Adam(model.parameters(), lr=3e-4)

TRAIN_DIR = 'data/npz_all/npz/tile/xla/train/'

filenames = [os.path.join(TRAIN_DIR, filename) for filename in os.listdir(TRAIN_DIR)]
num_epochs = 4
bs = 256

wandb.init(project='tpu_graphs')

itr = 0
for epoch in range(num_epochs):

	vl = valid_loss(model, bs, criterion)
	wandb.log({'valid_loss': vl})

	for filename in tqdm(filenames):
		graph_data = dict(np.load(filename, allow_pickle=True))
		node_feat = torch.from_numpy(graph_data['node_feat']).to(device)
		node_feat = (node_feat - 14.231035232543945) / 305.2548828125
		node_opcode = torch.from_numpy(graph_data['node_opcode']).to(device)
		edge_index = torch.from_numpy(graph_data['edge_index']).permute(1, 0).to(device)

		for trial_idx in range(0, len(graph_data['config_feat'][0]), bs):
			itr += 1
			next_idx = min(trial_idx + bs, len(graph_data['config_feat'][0]))

			config_feat = torch.from_numpy(graph_data['config_feat'][trial_idx:next_idx]).to(device)
			config_feat = (config_feat - 16.741966247558594) / 74.34544372558594
			config_runtime = torch.from_numpy(np.array([
				graph_data['config_runtime'][trial_idx:next_idx] / graph_data['config_runtime_normalizers'][trial_idx:next_idx]
			])).flatten().to(torch.float32).to(device)

			config_runtime = config_runtime / 8.203627220003426


			preds = model(config_feat)
			loss = torch.sqrt(criterion(preds.flatten(), config_runtime))

			loss.backward()
			optimizer.step()
			optimizer.zero_grad()

			wandb.log({'loss': loss.item()})

		if itr % 1500 == 0:
			plot_outputs_vs_predictions(config_runtime.cpu().detach(), preds.cpu().detach())
			plot_config(config_feat.cpu().detach())

	wandb.log({'epoch': epoch})

torch.save(model.state_dict(), 'model.pt')
