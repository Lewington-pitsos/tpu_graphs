from typing import Dict, Optional, List, Union, Tuple
import torch
from dataclasses import dataclass
import json
import math
import torch.nn as nn
from torch import Tensor
from transformers.modeling_outputs import BaseModelOutputWithPastAndCrossAttentions
from transformers.pytorch_utils import apply_chunking_to_forward
from transformers.activations import ACT2FN
from torch_geometric.nn import GCNConv, global_mean_pool

def count_parameters(model):
    num_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"The model has {num_parameters:,} trainable parameters")  # using ',' as a thousands separator


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


class SimpleModel(torch.nn.Module):
    def __init__(self, hidden_channels, graph_feats, hidden_dim):
        super().__init__()

        op_embedding_dim = 4  # I choose 4-dimensional embedding
        self.embedding = torch.nn.Embedding(120,  # 120 different op-codes
                                            op_embedding_dim,
                                           )
        assert len(hidden_channels) > 0
        in_channels = op_embedding_dim + 140
        self.convs = torch.nn.ModuleList()
        last_dim = hidden_channels[0]

        # Create a sequence of Graph Convolutional Network (GCN) layers
        self.convs.append(GCNConv(in_channels, hidden_channels[0]))
        for i in range(len(hidden_channels) - 1):
            self.convs.append(GCNConv(hidden_channels[i], hidden_channels[i+1]))
            last_dim = hidden_channels[i+1]
        self.convs.append(GCNConv(last_dim, graph_feats))

        # Define a sequential dense neural network
        self.dense = torch.nn.Sequential(nn.Linear(graph_feats + 24, 64),
                                         nn.ReLU(),
                                         nn.Linear(64, 64),
                                         nn.ReLU(),
                                         nn.Linear(64, 1),
                                        )

    def forward(self, x_cfg: Tensor, x_feat: Tensor, x_op: Tensor, edge_index: Tensor) -> Tensor:
        x = torch.cat([x_feat, self.embedding(x_op.long())], dim=1)
        for conv in self.convs:
            x = conv(x, edge_index).relu()

        x_graph = torch.mean(x, 0)

        x = torch.cat([x_cfg, x_graph.repeat((len(x_cfg), 1))], axis=1)
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
