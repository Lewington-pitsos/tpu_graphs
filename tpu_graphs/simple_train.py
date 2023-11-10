from tqdm import tqdm
import wandb
import torch
import os
from torch import nn, optim
import numpy as np
from .model import SimpleModel, count_parameters, ConfigDense
import matplotlib.pyplot as plt
from PIL import Image

if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')


model = ConfigDense(in_channels=24, out_channels=512, hidden=512)
# model = SimpleModel(hidden_channels=[128, 256, 512, 512, 1024], graph_feats=512)
count_parameters(model)

model.to(device)
criterion = nn.MSELoss()

optimizer = optim.Adam(model.parameters(), lr=3e-4)

TRAIN_DIR = 'data/npz_all/npz/tile/xla/valid/'
filenames = [os.path.join(TRAIN_DIR, filename) for filename in os.listdir(TRAIN_DIR)]
num_epochs = 10_000
bs = 32

def plot_config(config_feat, save_file):
    matrix = config_feat[:8].numpy()

    normalized_matrix = (matrix - matrix.min(axis=0)) / (matrix.ptp(axis=0))

    fig, ax = plt.subplots(figsize=(12, 3))  # Set the figure size
    for i in range(normalized_matrix.shape[1]):
        # Get the range for each column
        col_min = matrix[:, i].min()
        col_max = matrix[:, i].max()
        col_range = col_max - col_min

        if col_range > 0:  # Avoid division by zero
            # Normalize the column
            normalized_matrix[:, i] = (matrix[:, i] - col_min) / col_range
        else:
            # If the column range is 0 (i.e., all values are the same), set to midpoint (0.5)
            normalized_matrix[:, i] = 0.5

    # Now use matshow to display the independently normalized matrix
    cax = ax.matshow(normalized_matrix, cmap='viridis')

    # Loop over data dimensions and create text annotations with smaller font size.
    for i in range(normalized_matrix.shape[0]):
        for j in range(normalized_matrix.shape[1]):
            ax.text(j, i, f'{matrix[i, j]:.2f}', ha='center', va='center', color='w', fontsize=6)

    plt.savefig(save_file)

wandb.init(project='tpu_graphs')

for epoch in tqdm(range(num_epochs)):
    for filename in filenames:
        graph_data = dict(np.load(filename, allow_pickle=True))
        node_feat = torch.from_numpy(graph_data['node_feat']).to(device)
        node_opcode = torch.from_numpy(graph_data['node_opcode']).to(device)
        edge_index = torch.from_numpy(graph_data['edge_index']).permute(1, 0).to(device)

        for trial_idx in range(0, len(graph_data['config_feat'][0]), bs):
            next_idx = min(trial_idx + bs, len(graph_data['config_feat'][0]))

            config_feat = torch.from_numpy(graph_data['config_feat'][trial_idx:next_idx]).to(device)
            config_runtime = torch.from_numpy(np.array([
                graph_data['config_runtime'][trial_idx:next_idx] / graph_data['config_runtime_normalizers'][trial_idx:next_idx]
            ])).flatten().to(torch.float32).to(device)

            config_runtime = config_runtime / 8.203627220003426
            node_feat = (node_feat - 14.231035232543945) / 305.2548828125

            preds = model(config_feat)
            loss = torch.sqrt(criterion(preds.flatten(), config_runtime))

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            wandb.log({'loss': loss.item()})

            if epoch % 1000 == 0:

                max_plot = 16

                plt.figure(figsize=(10, 5))
                plt.scatter(config_runtime[:max_plot].cpu().detach(), preds[:max_plot].cpu().detach(), c=range(preds[:max_plot].shape[0]), cmap='viridis', alpha=0.7)
                plt.xlabel('Outputs')
                plt.ylabel('Predictions')
                plt.savefig('outputs_vs_predictions.png')
                plt.grid()
                plt.close()
                wandb.log({'outputs_vs_predictions': wandb.Image(Image.open('outputs_vs_predictions.png'))})

                plot_config(config_feat.cpu().detach(), 'config_feat.png')
                wandb.log({'input_plot': wandb.Image(Image.open('config_feat.png'))})
            break
        break

    wandb.log({'epoch': epoch})
