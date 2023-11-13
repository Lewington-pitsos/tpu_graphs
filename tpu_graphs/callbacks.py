import wandb
import tensorflow as tf
import tensorflow_gnn as tfgnn
import functools
import os
import numpy as np
import keras
from absl import app


class InputLogCB(keras.callbacks.Callback):
	def __init__(self, dataloader, n_epochs=5):
		super(InputLogCB, self).__init__()
		self.dataloader = dataloader
		self.n_epochs = n_epochs

	def on_epoch_begin(self, epoch, logs=None):
		if epoch % self.n_epochs == 0:
			for graph, _ in self.dataloader.take(1):

				config_feats = graph.node_sets['config'].features['feats'].cpu().numpy()
				node_feats = graph.node_sets['op'].features['feats'].cpu().numpy()
				# graph_feats = batch_graph_feats[0]

				wandb.log({
					"config_feature_image": wandb.Image(np.transpose(config_feats)),
					"config_feature_rows": config_feats.shape[0],
					"config_feature_cols": config_feats.shape[1],
					"config_feature_mean": config_feats.mean(),
					"config_feature_max": config_feats.max(),
					"config_feature_min": config_feats.min(),
					"config_feature_std": config_feats.std(),

					"graph_feature_image": wandb.Image(np.transpose(node_feats)),
					"graph_feature_rows": node_feats.shape[0],
					"graph_feature_cols": node_feats.shape[1],
					"graph_feature_mean": node_feats.mean(),
					"graph_feature_max": node_feats.max(),
					"graph_feature_min": node_feats.min(),
					"graph_feature_std": node_feats.std(),
				})
				# self.prev_runtimes = (tf.cast(graph.node_sets['config'].features['normalizers'], tf.float64) / graph.node_sets['config'].features['runtimes'])
				# self.prev_op = graph.node_sets['op'].features['op']
				# self.prev_feat = graph.node_sets['op'].features['feats']
				# self.prev_tile = graph.node_sets['g'].features['tile_id']
