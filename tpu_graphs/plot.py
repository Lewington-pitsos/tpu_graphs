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

	normalized_matrix = (matrix - matrix.min(axis=0)) / (matrix.ptp(axis=0) + 1)

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
	if matrix.size < 193:
		for i in range(normalized_matrix.shape[0]):
			for j in range(normalized_matrix.shape[1]):
				ax.text(j, i, f'{matrix[i, j]:.1f}', ha='center', va='center', color='w', fontsize=6)

	plt.savefig(save_file)
	plt.close(fig)
	plt.close()
	img = Image.open(save_file)
	wandb.log({name: wandb.Image(img)})
	img.close()


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
