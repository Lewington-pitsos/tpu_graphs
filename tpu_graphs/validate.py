import os
import torch
import torch.nn as nn
import numpy as np
from .score import speed_score
from .model import ConfigDense, GraphModel
from .pt_loader	import file_data

def file_preds(filename, model, bs, device):
	node_feat, node_opcode, edge_index, config_feat, config_runtime = file_data(filename, device)

	all_preds = []
	all_runtimes = []
	with torch.no_grad():
		for trial_idx in range(0, len(config_runtime), bs):
			next_idx = trial_idx + bs

			preds = model(config_feat[trial_idx:next_idx], node_feat, node_opcode, edge_index)
			all_preds.extend(list(preds.flatten().cpu().detach().numpy()))
			all_runtimes.extend(list(config_runtime[trial_idx:next_idx].cpu().detach().numpy()))

	return np.array(all_preds), np.array(all_runtimes)

def valid_loss(model, criterion, device, K=5):
	VALID_DIR = 'data/npz_all/npz/tile/xla/valid/'
	valid_filenames = [os.path.join(VALID_DIR, filename) for filename in os.listdir(VALID_DIR)]

	losses = []
	rand_scores = []
	perfect_scores = []
	model_scores = []
	for filename in valid_filenames:
		preds, config_runtime = file_preds(filename, model, 512, device)
		preds, config_runtime = torch.from_numpy(preds), torch.from_numpy(config_runtime)
		loss = torch.sqrt(criterion(preds, config_runtime))
		losses.append(loss.mean().item())

		config_runtime = config_runtime.numpy()
		preds = preds.numpy()

		perfect_preds = np.argsort(config_runtime)
		perfect_score = speed_score(config_runtime, perfect_preds, K)
		perfect_scores.append(perfect_score)

		np.random.shuffle(perfect_preds)
		rand_score = speed_score(config_runtime, perfect_preds, K)
		rand_scores.append(rand_score)

		predicted_idx = np.argsort(preds)
		model_score = speed_score(config_runtime, predicted_idx, K)
		model_scores.append(model_score)

	return {
		'loss': np.mean(losses),
		'perfect_score': np.mean(perfect_scores),
		'random_score': np.mean(rand_scores),
		'model_score': np.mean(model_scores)
	}


if __name__ == '__main__':
	if torch.cuda.is_available():
		device = torch.device('cuda')
	else:
		device = torch.device('cpu')

	VALID_DIR = 'data/npz_all/npz/tile/xla/valid/'
	valid_files = [os.path.join(VALID_DIR, filename) for filename in os.listdir(VALID_DIR)]

	perfect_scores = []
	rand_scores = []
	linear_scores = []

	# model = ConfigDense(in_channels=24, out_channels=512, hidden=512)
	model = GraphModel(hidden_channels=[128, 256, 512, 512, 1024], graph_feats=512)

	model.load_state_dict(torch.load('model.pt'))
	model.to(device)
	model.eval()

	# RMSE Loss
	criterion = lambda x, y: torch.sqrt(nn.MSELoss()(x, y))

	print(valid_loss(model, torch.nn.MSELoss(), device))


	# perfect score 1.0
	# random score 0.49241369138166075
	# config score 0.6849965446934841
