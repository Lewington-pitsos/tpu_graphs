import uuid
import numpy as np
import pandas as pd
import torch
import numpy as np
from .model import ConfigDense
from .conly import file_preds

TEST_DIR = 'data/npz_all/npz/tile/xla/test/'

def generate_ordering(identifier, model):
    if "tile:xla:" in identifier:
        filename = TEST_DIR + identifier.split(":")[-1] + '.npz'
        y_pred = model(filename)

        indices = np.argsort(y_pred)

        return ";".join([str(i) for i in list(indices[:5])])
    else:
        return ";".join(map(str, range(5)))


def submit(model, example_file, output_file=None):
	if output_file is None:
		output_file = f'submissions/{str(uuid.uuid4())}.csv'

	print('submitting to', output_file)

	df = pd.read_csv(example_file)

	baseline_submission = pd.DataFrame({
		'ID': df['ID'],
		'TopConfigs': [generate_ordering(i, model) for i in df['ID'].values]
	})

	baseline_submission.to_csv(output_file, index=False)

def dense_model_fn(device):
	model = ConfigDense(in_channels=24, out_channels=512, hidden=512)
	model.load_state_dict(torch.load('model.pt'))
	model.to(device)

	def model_fn(filename):
		preds, _ = file_preds(filename, model, 256, device)
		return preds

	return model_fn

if torch.cuda.is_available():
	device = torch.device('cuda')
else:
	device = torch.device('cpu')

model = dense_model_fn(device)

submit(model, 'data/sample_submission.csv')
