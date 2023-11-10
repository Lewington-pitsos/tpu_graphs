import os
import torch
import numpy as np
from .score import speed_score
from .model import ConfigDense

def file_preds(filename, model, bs, device):
	graph_data = dict(np.load(filename, allow_pickle=True))
	node_feat = torch.from_numpy(graph_data['node_feat']).to(device)
	node_feat = (node_feat - 14.231035232543945) / 305.2548828125
	node_opcode = torch.from_numpy(graph_data['node_opcode']).to(device)
	edge_index = torch.from_numpy(graph_data['edge_index']).permute(1, 0).to(device)

	all_preds = []
	all_runtimes = []
	with torch.no_grad():
		for trial_idx in range(0, len(graph_data['config_runtime']), bs):
			next_idx = min(trial_idx + bs, len(graph_data['config_runtime']))

			config_feat = torch.from_numpy(graph_data['config_feat'][trial_idx:next_idx]).to(device)
			config_feat = (config_feat - 16.741966247558594) / 74.34544372558594
			config_runtime = torch.from_numpy(np.array([
				graph_data['config_runtime'][trial_idx:next_idx] / graph_data['config_runtime_normalizers'][trial_idx:next_idx]
			])).flatten().to(torch.float32).to(device)


			config_runtime = config_runtime / 8.203627220003426

			preds = model(config_feat)
			all_preds.extend(list(preds.flatten().cpu().detach().numpy()))
			all_runtimes.extend(list(config_runtime.cpu().detach().numpy()))

	return np.array(all_preds), np.array(all_runtimes)


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

	model = ConfigDense(in_channels=24, out_channels=512, hidden=512)
	model.load_state_dict(torch.load('model.pt'))
	model.to(device)
	K = 6
	for filename in valid_files:
		unnormalised_runtime = np.load(filename, allow_pickle=True)['config_runtime']
		runtime_normalisers = np.load(filename, allow_pickle=True)['config_runtime_normalizers']
		runtimes = unnormalised_runtime / runtime_normalisers

		perfect_preds = np.argsort(runtimes)
		perfect_score = speed_score(runtimes, perfect_preds, K)
		perfect_scores.append(perfect_score)

		# randomize the indices
		np.random.shuffle(perfect_preds)
		rand_score = speed_score(runtimes, perfect_preds, K)
		rand_scores.append(rand_score)

		predicted_runtimes, _ = file_preds(filename, model, 256, device)
		predicted_idx = np.argsort(predicted_runtimes)
		linear_score = speed_score(runtimes, predicted_idx, K)
		linear_scores.append(linear_score)

	print("perfect score", np.mean(perfect_scores))
	print("random score", np.mean(rand_scores))
	print("config score", np.mean(linear_scores))
