import wandb
import tensorflow as tf
import numpy as np
import keras
from tpu_graphs.plot import plot_outputs_vs_predictions, plot_config

class BatchLossLogger(tf.keras.callbacks.Callback):
		def __init__(self):
				super().__init__()
				self.total_loss = 0.0
				self.total_batches = 0

		def on_epoch_begin(self, epoch, logs=None):
				self.total_loss = 0.0
				self.total_batches = 0

		def on_train_batch_end(self, batch, logs=None):
				logs = logs or {}
				current_avg_loss = logs.get('loss')

				if current_avg_loss is not None:
						current_total_loss = current_avg_loss * (self.total_batches + 1)
						batch_loss = current_total_loss - self.total_loss

						self.total_loss = current_total_loss
						self.total_batches += 1

						wandb.log({"raw_loss": batch_loss})

class InputLogCB(keras.callbacks.Callback):
	def __init__(self, dataloader, n_epochs=5):
		super(InputLogCB, self).__init__()
		self.dataloader = dataloader
		self.graph, self.y = next(iter(dataloader))
		self.batch_size = int(self.graph.node_sets['config'].sizes[0].numpy())

		self.config_feats = self.graph.node_sets['config'].features['feats'].cpu().numpy()
		self.node_feats = self.graph.node_sets['op'].features['feats'].cpu().numpy()
		self.runtimes = (self.graph.node_sets['config'].features['runtimes'][:self.batch_size] / tf.cast(self.graph.node_sets['config'].features['normalizers'][:self.batch_size], tf.float64))

		self.n_epochs = n_epochs

		plot_config(self.config_feats, 'config_features')
		plot_config(self.node_feats, 'graph_features')


		wandb.log({
			"config_feature_rows": self.config_feats.shape[0],
			"config_feature_cols": self.config_feats.shape[1],
			"config_feature_mean": self.config_feats.mean(),
			"config_feature_max": self.config_feats.max(),
			"config_feature_min": self.config_feats.min(),
			"config_feature_std": self.config_feats.std(),

			"graph_feature_image": wandb.Image(np.transpose(self.node_feats)),
			"graph_feature_rows": self.node_feats.shape[0],
			"graph_feature_cols": self.node_feats.shape[1],
			"graph_feature_mean": self.node_feats.mean(),
			"graph_feature_max": self.node_feats.max(),
			"graph_feature_min": self.node_feats.min(),
			"graph_feature_std": self.node_feats.std(),
		})


	def on_epoch_begin(self, epoch, logs=None):
		if epoch % self.n_epochs == 0:
			with tf.GradientTape(persistent=False, watch_accessed_variables=False):
				outputs = self.model.forward(self.graph, self.batch_size)

			plot_outputs_vs_predictions(self.runtimes, outputs[0])


class PredictionCB(keras.callbacks.Callback):
		def __init__(self, batch_to_log):
				super(PredictionCB, self).__init__()
				self.batch_to_log = batch_to_log  # The batch number you want to log.

		def on_batch_end(self, batch, logs=None):
				if batch == self.batch_to_log:
						x, y = self.validation_data[0], self.validation_data[1]
						predictions = self.model.predict(x)
						print("\nLogging Info for Batch {}: ".format(self.batch_to_log))
						print("Actual outputs: ", y)
						print("Predictions: ", predictions)
