import os
import torch
import numpy as np
from .score import speed_score
from .model import ConfigDense
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
	# perfect score 1.0
	# random score 0.49241369138166075
	# config score 0.6849965446934841
