import torch
import torch.nn as nn
from torch import Tensor
from torch_geometric.nn import GCNConv, global_mean_pool

def count_parameters(model):
	num_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
	print(f"The model has {num_parameters:,} trainable parameters")  # using ',' as a thousands separator
	return num_parameters

class ConfigDense(nn.Module):
	def __init__(self, in_channels, out_channels, hidden):
		super(ConfigDense, self).__init__()
		self.fc = nn.Linear(in_channels, out_channels)
		self.fc2 = nn.Linear(out_channels, hidden)
		self.fc3 = nn.Linear(hidden, hidden)
		self.fc4 = nn.Linear(hidden, 1)

		self.activation = nn.ReLU()

	def forward(self, config: Tensor, node_features: Tensor, opcodes: Tensor, edge_index: Tensor):
		x = self.activation(self.fc(config))
		x = self.activation(self.fc2(x))
		x = self.activation(self.fc3(x))
		x = self.fc4(x)

		return x

class Opcodes(nn.Module):
	def __init__(self, in_channels, out_channels, hidden, op_embedding_dim=128):
		super(Opcodes, self).__init__()

		self.embedding = torch.nn.Embedding(120, op_embedding_dim )

		self.emb_fc = nn.Linear(op_embedding_dim, hidden)
		self.emb_fc2 = nn.Linear(hidden, hidden)


		in_channels = hidden + 24


		self.fc = nn.Linear(in_channels, out_channels)
		self.fc2 = nn.Linear(out_channels, hidden)
		self.fc3 = nn.Linear(hidden, 1)
		self.fc3.bias.data = torch.tensor([0.607150242])

		self.activation = nn.ReLU()

	def forward(self, config: Tensor, node_features: Tensor, opcodes: Tensor, edge_index: Tensor):
		x = self.embedding(opcodes.long()) # (n_nodes, op_embedding_dim)

		x = self.activation(self.emb_fc(x))
		x = self.activation(self.emb_fc2(x))

		x = torch.mean(x, 0) # (hidden)

		x = x.repeat((config.shape[0], 1)) # (batch_size, hidden)

		x = torch.cat([x, config], axis=1)

		x = self.activation(self.fc(x))
		x = self.activation(self.fc2(x))
		x = self.fc3(x)

		return x

class FeatureConv(nn.Module):
	def __init__(self, in_channels, out_channels, kernel_width):
		super(FeatureConv, self).__init__()
		self.conv = nn.Conv1d(
			in_channels=in_channels,
			out_channels=out_channels,
			kernel_size=(kernel_width,)
		  )


	def forward(self, x):
		return self.conv(x)

class OneDModel(nn.modules.Module):
	def __init__(self, num_features):
		super(OneDModel, self).__init__()
		self.conv1 = FeatureConv(in_channels=num_features, out_channels=256, kernel_width=5)
		self.conv2 = FeatureConv(in_channels=256, out_channels=512, kernel_width=5)
		self.conv3 = FeatureConv(in_channels=512, out_channels=512, kernel_width=3)
		self.conv4 = FeatureConv(in_channels=512, out_channels=1024, kernel_width=3)
		self.conv5 = FeatureConv(in_channels=1024, out_channels=1024, kernel_width=3)


		self.fc1 = nn.Linear(1048, 1048)
		self.fc2 = nn.Linear(1048, 512)
		self.fc3 = nn.Linear(512, 512)

		self.fc4 = nn.Linear(512, 256)
		self.fc5 = nn.Linear(256, 1)

		self.activation = nn.Tanh()

	def forward(self, node_feat, config_feat):
		node_feat = node_feat.permute(0, 2, 1)

		node_feat = self.conv1(node_feat)
		node_feat = self.activation(node_feat)

		node_feat = self.conv2(node_feat)
		node_feat = self.activation(node_feat)

		node_feat = self.conv3(node_feat)
		node_feat = self.activation(node_feat)

		node_feat = self.conv4(node_feat)
		node_feat = self.activation(node_feat)

		node_feat = self.conv5(node_feat)
		node_feat = self.activation(node_feat)

		node_feat = torch.mean(node_feat, dim=2)
		combined = torch.concat([node_feat, config_feat], dim=1)

		combined = self.activation(self.fc1(combined))
		combined = self.activation(self.fc2(combined))
		combined = self.activation(self.fc3(combined))
		combined = self.activation(self.fc4(combined))

		combined = self.fc5(combined)

		return combined


class GraphModel(torch.nn.Module):
	def __init__(self, hidden_channels, graph_feats, op_embedding_dim=128):
		super().__init__()

		self.embedding = torch.nn.Embedding(120, op_embedding_dim )
		in_channels = op_embedding_dim + 140
		self.convs = torch.nn.ModuleList()

		self.convs.append(GCNConv(in_channels, hidden_channels[0]))
		for i in range(len(hidden_channels) - 1):
			self.convs.append(GCNConv(hidden_channels[i], hidden_channels[i+1]))
		self.convs.append(GCNConv(hidden_channels[-1], graph_feats))


		final_layer = nn.Linear(graph_feats, 1)
		final_layer.bias.data = torch.tensor([0.607150242])

		self.dense = torch.nn.Sequential(
			nn.Linear(graph_feats + 24, graph_feats),
			nn.ReLU(),
			nn.Linear(graph_feats, graph_feats),
			nn.ReLU(),
			final_layer
		)

	def forward(self, config: Tensor, node_features: Tensor, opcodes: Tensor, edge_index: Tensor) -> Tensor:
		x = torch.cat([node_features, self.embedding(opcodes.long())], dim=1)

		for conv in self.convs:
			x = conv(x, edge_index).relu()

		x_graph = torch.mean(x, 0)

		x = torch.cat([config, x_graph.repeat((len(config), 1))], axis=1)
		x = torch.flatten(self.dense(x))

		return x


class GCN(torch.nn.Module):
	def __init__(self, in_channels, layer_config):
		super().__init__()

		conv_layers = []


		for config in layer_config:
			conv_layers.append(GCNConv(in_channels, config['out_channels']))
			in_channels = config['out_channels']

		self.conv_layers = nn.ModuleList(conv_layers)

		self.linear = nn.Linear(in_channels, 128)
		self.linear2 = nn.Linear(128, 1)

	def forward(self, x: Tensor, edge_index: Tensor, batch_map: Tensor) -> Tensor:
		for conv in self.conv_layers:
			x = conv(x, edge_index).relu()

		x = global_mean_pool(x, batch_map)

		x = self.linear(x).relu()
		x = self.linear2(x)

		return x

# if __name__ == '__main__':
#     model = OneDModel(num_features=141)

#     input_tensor = torch.randn(2, 27, 141)
#     config_feat = torch.randn(2, 24)

#     output = model(input_tensor, config_feat)

#     print(output)
