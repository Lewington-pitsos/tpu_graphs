import numpy as np
import wandb
import matplotlib.pyplot as plt
from PIL import Image

def plot_outputs_vs_predictions(config_runtime, preds):
	max_plot = 16
	plt.figure(figsize=(10, 5))
	plt.scatter(config_runtime[:max_plot].cpu().detach(), preds[:max_plot].cpu().detach(), c=range(preds[:max_plot].shape[0]), cmap='viridis', alpha=0.7)
	plt.xlabel('Outputs')
	plt.ylabel('Predictions')
	plt.grid()
	plt.savefig('outputs_vs_predictions.png')
	plt.close()
	img = Image.open('outputs_vs_predictions.png')
	wandb.log({'outputs_vs_predictions': wandb.Image(img)})
	img.close()

def plot_config(config_feat, name):
	save_file = 'tmp.png'
	matrix = config_feat[:8].numpy()

	col_min = matrix.min(axis=0)
	col_ptp = matrix.ptp(axis=0)

	normalized_matrix = (matrix - col_min) / (col_ptp + (col_ptp == 0))

	fig, ax = plt.subplots(figsize=(12, 3))
	cax = ax.matshow(normalized_matrix, cmap='viridis')

	if normalized_matrix.size <= 192:  # Adjust this threshold based on your needs
		for i in range(normalized_matrix.shape[0]):
			for j in range(normalized_matrix.shape[1]):
				ax.text(j, i, f'{matrix[i, j]}', ha='center', va='center', color='w', fontsize=6)

	plt.savefig(save_file)
	plt.close(fig)

	with Image.open(save_file) as img:
		wandb.log({name: wandb.Image(img)})

def plot_opcodes(opcodes):
	save_file = 'tmp.png'

	# Convert the list of integers to a numpy array and normalize it
	matrix = np.array(opcodes)
	normalized_matrix = (matrix - matrix.min()) / (matrix.ptp() + 1)

	# Create a figure with appropriate size
	fig, ax = plt.subplots(figsize=(18, 8))

	# Reshape the normalized matrix to a 2D array with 1 row
	matrix_2d = normalized_matrix.reshape(1, -1)

	# Display the matrix
	cax = ax.matshow(matrix_2d, cmap='viridis')


	for i, val in enumerate(matrix):
		ax.text(i, 0, f'{val}', ha='center', va='center', color='w', fontsize=6)

	# Save the figure
	plt.savefig(save_file)
	plt.close(fig)

	# Open the saved image and log it with wandb
	img = Image.open(save_file)
	wandb.log({'opcodes': wandb.Image(img)})
	img.close()
