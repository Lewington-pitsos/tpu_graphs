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

        # Get graph features
        x = torch.cat([x_feat, self.embedding(x_op)], dim=1)
        # Pass data through convolutional layers
        for conv in self.convs:
            x = conv(x, edge_index).relu()

        # Get 1D graph embedding using average pooling
        x_graph = torch.mean(x, 0)

        # Combine graph data with config data
        x = torch.cat([x_cfg, x_graph.repeat((len(x_cfg), 1))], axis=1)
        # Pass the combined data through the dense neural network
        x = torch.flatten(self.dense(x))

        # Standardize the output
        x = (x - torch.mean(x)) / (torch.std(x) + 1e-5)
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

NODE_OP_CODES = 120
NODE_FEATS = 140
CONFIG_FEATS = 24
NODE_CONFIG_FEATS = 18

@dataclass
class GraphConfig:
    num_hidden_layers: int = 8
    hidden_size: int = 256
    num_attention_heads: int = 16
    intermediate_size: int = 64
    chunk_size_feed_forward: int = 64
    attention_probs_dropout_prob: float = 0.0
    max_position_embeddings: int = 512
    hidden_dropout_prob: float = 0.0
    layer_norm_eps: float = 1e-12
    hidden_act: str = 'gelu'
    initializer_range: float = 0.02
    output_hidden_states: bool = False
    output_attentions: bool = False
    gradient_checkpointing: bool = False
    margin: float = 0.1
    number_permutations: int = 10

    def __post_init__(self):
        self.embedding_size = self.hidden_size

    def validate(self):
        if self.hidden_size % self.num_attention_heads != 0 and not hasattr(self, "embedding_size"):
            raise ValueError(
                f"The hidden size ({self.hidden_size}) is not a multiple of the number of attention "
                f"heads ({self.num_attention_heads})"
            )

    def save_config(self, path):
        config = dict(self)
        with open(path, 'w') as f:
            json.dump(config, f)

    @classmethod
    def load_config(cls, path):
        with open(path, 'r') as f:
            config = json.load(f)
        return cls(**config)

# Modified from https://github.com/huggingface/transformers/blob/main/src/transformers/models/bert/modeling_bert.py
class BertEncoder(nn.Module):
    def __init__(self, config:GraphConfig):
        super().__init__()
        self.config = config
        self.layer = nn.ModuleList([BertLayer(config) for _ in range(config.num_hidden_layers)])
        self.gradient_checkpointing = False

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.FloatTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        output_attentions: Optional[bool] = False,
        output_hidden_states: Optional[bool] = False,
        return_dict: Optional[bool] = True,
    ) -> Union[Tuple[torch.Tensor], BaseModelOutputWithPastAndCrossAttentions]:
        all_hidden_states = () if output_hidden_states else None
        all_self_attentions = () if output_attentions else None

        for i, layer_module in enumerate(self.layer):
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            layer_head_mask = head_mask #DONE: Same Head Mask for all layers

            if self.gradient_checkpointing and self.training:

                def create_custom_forward(module):
                    def custom_forward(*inputs):
                        return module(*inputs,  output_attentions)

                    return custom_forward

                layer_outputs = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(layer_module),
                    hidden_states,
                    attention_mask,
                    layer_head_mask,
                )
            else:
                layer_outputs = layer_module(
                    hidden_states,
                    attention_mask,
                    layer_head_mask,
                    output_attentions,
                )

            hidden_states = layer_outputs[0]
            if output_attentions:
                all_self_attentions = all_self_attentions + (layer_outputs[1],)

        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        if not return_dict:
            return tuple(
                v
                for v in [
                    hidden_states,
                    all_hidden_states,
                    all_self_attentions,
                ]
                if v is not None
            )
        return BaseModelOutputWithPastAndCrossAttentions(
            last_hidden_state=hidden_states,
            past_key_values=None,
            hidden_states=all_hidden_states,
            attentions=all_self_attentions,
            cross_attentions=None,
        )


class BertLayer(nn.Module):
    def __init__(self, config:GraphConfig):
        super().__init__()
        self.chunk_size_feed_forward = config.chunk_size_feed_forward
        self.seq_len_dim = 1
        self.attention = BertAttention(config)
        self.intermediate = BertIntermediate(config)
        self.output = BertOutput(config)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.FloatTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        output_attentions: Optional[bool] = False,
    ) -> Tuple[torch.Tensor]:
        # decoder uni-directional self-attention cached key/values tuple is at positions 1,2
        self_attention_outputs = self.attention(
            hidden_states,
            attention_mask,
            head_mask,
            output_attentions=output_attentions,
        )
        attention_output = self_attention_outputs[0]
        outputs = self_attention_outputs[1:]  # add self attentions if we output attention weights
        layer_output = apply_chunking_to_forward(
            self.feed_forward_chunk, self.chunk_size_feed_forward, self.seq_len_dim, attention_output
        )
        outputs = (layer_output,) + outputs


        return outputs

    def feed_forward_chunk(self, attention_output):
        intermediate_output = self.intermediate(attention_output)
        layer_output = self.output(intermediate_output, attention_output)
        return layer_output

class BertIntermediate(nn.Module):
    def __init__(self, config:GraphConfig):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.intermediate_size)
        if isinstance(config.hidden_act, str):
            self.intermediate_act_fn = ACT2FN[config.hidden_act]
        else:
            self.intermediate_act_fn = config.hidden_act

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.dense(hidden_states)
        hidden_states = self.intermediate_act_fn(hidden_states)
        return hidden_states

class BertOutput(nn.Module):
    def __init__(self, config:GraphConfig):
        super().__init__()
        self.dense = nn.Linear(config.intermediate_size, config.hidden_size)
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states: torch.Tensor, input_tensor: torch.Tensor) -> torch.Tensor:
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states

class BertAttention(nn.Module):
    def __init__(self, config:GraphConfig, position_embedding_type=None):
        super().__init__()
        self.self = BertSelfAttention(config, position_embedding_type=position_embedding_type)
        self.output = BertSelfOutput(config)
        self.pruned_heads = set()


    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.FloatTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        output_attentions: Optional[bool] = False,
    ) -> Tuple[torch.Tensor]:
        self_outputs = self.self(
            hidden_states,
            attention_mask,
            head_mask,
            output_attentions,
        )
        attention_output = self.output(self_outputs[0], hidden_states)
        outputs = (attention_output,) + self_outputs[1:]  # add attentions if we output them
        return outputs


class BertSelfAttention(nn.Module):
    def __init__(self, config:GraphConfig, position_embedding_type=None):
        super().__init__()
        if config.hidden_size % config.num_attention_heads != 0 and not hasattr(config, "embedding_size"):
            raise ValueError(
                f"The hidden size ({config.hidden_size}) is not a multiple of the number of attention "
                f"heads ({config.num_attention_heads})"
            )

        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = nn.Linear(config.hidden_size, self.all_head_size)
        self.key = nn.Linear(config.hidden_size, self.all_head_size)
        self.value = nn.Linear(config.hidden_size, self.all_head_size)

        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)
        self.position_embedding_type = position_embedding_type or getattr(
            config, "position_embedding_type", "absolute"
        )
        if self.position_embedding_type == "relative_key" or self.position_embedding_type == "relative_key_query":
            self.max_position_embeddings = config.max_position_embeddings
            self.distance_embedding = nn.Embedding(2 * config.max_position_embeddings - 1, self.attention_head_size)


    def transpose_for_scores(self, x: torch.Tensor) -> torch.Tensor:
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.FloatTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        output_attentions: Optional[bool] = False,
    ) -> Tuple[torch.Tensor]:

        mixed_query_layer = self.query(hidden_states)
        key_layer = self.transpose_for_scores(self.key(hidden_states))
        value_layer = self.transpose_for_scores(self.value(hidden_states))
        query_layer = self.transpose_for_scores(mixed_query_layer)


        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))

        if self.position_embedding_type == "relative_key" or self.position_embedding_type == "relative_key_query":
            query_length, key_length = query_layer.shape[2], key_layer.shape[2]
            position_ids_l = torch.arange(query_length, dtype=torch.long, device=hidden_states.device).view(-1, 1)
            position_ids_r = torch.arange(key_length, dtype=torch.long, device=hidden_states.device).view(1, -1)
            distance = position_ids_l - position_ids_r

            positional_embedding = self.distance_embedding(distance + self.max_position_embeddings - 1)
            positional_embedding = positional_embedding.to(dtype=query_layer.dtype)  # fp16 compatibility

            if self.position_embedding_type == "relative_key":
                relative_position_scores = torch.einsum("bhld,lrd->bhlr", query_layer, positional_embedding)
                attention_scores = attention_scores + relative_position_scores
            elif self.position_embedding_type == "relative_key_query":
                relative_position_scores_query = torch.einsum("bhld,lrd->bhlr", query_layer, positional_embedding)
                relative_position_scores_key = torch.einsum("bhrd,lrd->bhlr", key_layer, positional_embedding)
                attention_scores = attention_scores + relative_position_scores_query + relative_position_scores_key

        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        if attention_mask is not None:
            # Apply the attention mask is (precomputed for all layers in BertModel forward() function)
            attention_scores = attention_scores + attention_mask

        # Normalize the attention scores to probabilities.
        attention_probs = nn.functional.softmax(attention_scores, dim=-1)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs = self.dropout(attention_probs)

        # Mask heads if we want to
        if head_mask is not None:
            attention_probs = attention_probs * head_mask #DONE: Same Head Mask for all Heads

        context_layer = torch.matmul(attention_probs, value_layer)

        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(new_context_layer_shape)

        outputs = (context_layer, attention_probs) if output_attentions else (context_layer,)

        return outputs


class BertSelfOutput(nn.Module):
    def __init__(self, config:GraphConfig):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states: torch.Tensor, input_tensor: torch.Tensor) -> torch.Tensor:
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


class NodeEncoder(nn.Module):

    def __init__(self, config:GraphConfig):
        super().__init__()
        self.node_opcode_embeddings = nn.Embedding(NODE_OP_CODES+1 , config.embedding_size, padding_idx=NODE_OP_CODES)
        self.linear = nn.Linear(NODE_FEATS, config.embedding_size, bias=False)
        self.layer_norm = nn.LayerNorm(config.embedding_size, eps=config.layer_norm_eps)


    def forward(self,
                node_opcode: torch.Tensor,
                node_feat: torch.Tensor
                ) -> torch.Tensor:
        opcode_embeddings = self.node_opcode_embeddings(node_opcode)
        node_feats =  self.linear(node_feat)
        features = opcode_embeddings + node_feats
        features = self.layer_norm(features)
        return features


class BertNodeEncoder(nn.Module):

    def __init__(self, config:GraphConfig) -> None:
        super().__init__()
        self.config = config
        self.node_embeddings = NodeEncoder(config)
        self.node_encoder = BertEncoder(config)

    def forward(self,
                node_opcode: torch.Tensor,
                node_feat: torch.Tensor,
                edges_adjecency: torch.Tensor,
                node_attn_mask: torch.Tensor
                ):
        node_embeddings = self.node_embeddings(node_opcode, node_feat)
        node_attn_mask = node_attn_mask.unsqueeze(1).unsqueeze(-1)
        node_encoder_outputs = self.node_encoder(node_embeddings,
                                                 attention_mask=node_attn_mask,
                                                 head_mask=edges_adjecency.unsqueeze(0).repeat(self.config.num_hidden_layers, 1, 1, 1).unsqueeze(2),
                                                 output_attentions=True)
        return node_encoder_outputs

def transform_node_positional_embeddings(embeddings_output:torch.Tensor,
                                         node_config_ids:torch.Tensor,
                                         num_nodes:int
                                         ) -> torch.Tensor:
    bs, num_configs, _, dim = embeddings_output.shape
    idxs = node_config_ids.unsqueeze(1).repeat(1,num_configs,1)
    zeros = torch.zeros(bs, num_configs, num_nodes, dim, device=embeddings_output.device, dtype=embeddings_output.dtype)
    idxs = idxs.unsqueeze(-1).repeat(1,1,1,dim)
    zeros.scatter_reduce_(2, idxs, embeddings_output, reduce='sum')
    return zeros

class NodeFeatEmbeddings(nn.Module):
    def __init__(self, config:GraphConfig):
        super().__init__()
        self.config = config
        self.node_feat_embeddings = nn.Linear(NODE_CONFIG_FEATS + CONFIG_FEATS, config.embedding_size, bias=False)
        self.layer_norm = nn.LayerNorm(config.embedding_size, eps=config.layer_norm_eps)

    def forward(self, node_config_feat: torch.Tensor, node_config_ids: torch.Tensor, num_nodes:int) -> torch.Tensor:
        node_config_feat_embeddings = self.node_feat_embeddings(node_config_feat)
        node_config_feat_embeddings = self.layer_norm(node_config_feat_embeddings)
        node_config_feat_embeddings = transform_node_positional_embeddings(node_config_feat_embeddings, node_config_ids, num_nodes)
        return node_config_feat_embeddings

class BertGraphEncoder(nn.Module):
    def __init__(self, config:GraphConfig) -> None:
        super().__init__()
        self.config = config
        self.node_embeddings = NodeEncoder(config)
        self.node_encoder = BertEncoder(config)
        self.node_feat_embeddings = NodeFeatEmbeddings(config)

    def forward(self,
                node_opcode: torch.Tensor, # (bs, num_nodes)
                node_feat: torch.Tensor, # (bs, num_nodes, num_node_feats)
                edges_adjecency: torch.Tensor, # (bs, num_nodes, num_nodes)
                node_attn_mask: torch.Tensor, # (bs, num_nodes)
                node_config_feat: torch.Tensor, # (bs, num_configs, num_config_nodes, num_node_feats)
                node_config_ids: torch.Tensor, # (bs, num_configs, num_config_nodes)
                ):
        bs, num_nodes = node_opcode.shape
        num_configs = node_config_feat.shape[1]
        node_embeddings = self.node_embeddings(node_opcode, node_feat)
        node_config_feat_embeddings = self.node_feat_embeddings(node_config_feat, node_config_ids, num_nodes)

        node_embeddings = node_embeddings.unsqueeze(1).repeat(1, num_configs, 1, 1)
        node_embeddings += node_config_feat_embeddings
        node_attn_mask = node_attn_mask.unsqueeze(1).repeat(1, num_configs, 1)
        node_embeddings = node_embeddings.reshape(bs *num_configs, num_nodes, -1)
        node_attn_mask = node_attn_mask.reshape(bs *num_configs, num_nodes)
        node_attn_mask = node_attn_mask.unsqueeze(1).unsqueeze(-1)
        edges_adjecency = edges_adjecency.unsqueeze(1).repeat(1, num_configs, 1, 1).reshape(bs *num_configs, num_nodes, num_nodes)
        edges_adjecency = edges_adjecency.unsqueeze(1)


        node_encoder_outputs = self.node_encoder(node_embeddings,
                                                 attention_mask=node_attn_mask,
                                                 head_mask=edges_adjecency,
                                                 output_attentions=True)

        return node_encoder_outputs.last_hidden_state.reshape(bs, num_configs, num_nodes, -1)


class MultiElementRankLoss(nn.Module):
    """
    Loss function that compares the output of the model with the output of the model with a permutation of the elements
    """

    def __init__(self, margin:float=0.0, number_permutations:int = 1) -> None:
        super().__init__()
        self.loss_fn = torch.nn.MarginRankingLoss(margin=margin, reduction = 'none')
        self.number_permutations = number_permutations

    def calculate_rank_loss(self,
                            outputs: torch.Tensor,
                            config_runtime: torch.Tensor,
                            config_idxs: torch.Tensor
                            ):
        """
        Generates a permutation of the predictions and targets and calculates the loss MarginRankingLoss against the permutation
        Args:
            outputs: Tensor of shape (bs, seq_len) with the outputs of the model
            config_runtime: Tensor of shape (bs, seq_len) with the runtime of the model
            config_mask: Tensor of shape (bs, seq_len) with 1 in the positions of the elements
            and 0 in the positions of the padding
        Returns:
            loss: Tensor of shape (bs, seq_len) with the loss for each element in the batch
        """
        bs, num_configs = outputs.shape
        permutation = torch.randperm(num_configs)
        permuted_idxs = config_idxs[:, permutation]
        # We mask those cases where we compare the same configuration
        config_mask = torch.where(config_idxs != permuted_idxs, 1, 0)
        permuted_runtime = config_runtime[:, permutation]
        labels = 2*((config_runtime - permuted_runtime) > 0) -1
        permuted_output = outputs[:, permutation]
        loss = self.loss_fn(outputs.view(-1,1), permuted_output.view(-1,1), labels.view(-1,1))
        loss = loss.view(bs, num_configs) * config_mask
        return loss.mean()


    def forward(self,
                outputs: torch.Tensor,
                config_runtime: torch.Tensor,
                config_idxs: torch.Tensor
                ):
        loss = 0
        for _ in range(self.number_permutations):
            loss += self.calculate_rank_loss(outputs, config_runtime, config_idxs)
        return loss/ self.number_permutations

class GraphEncoder(nn.Module):

    config_class = GraphConfig

    def __init__(self, config:GraphConfig):
        super().__init__()
        self.config = config
        self.node_encoder = BertGraphEncoder(config)
        self.head = nn.Linear(config.hidden_size, 1)
        self.loss_fn = MultiElementRankLoss(margin=config.margin, number_permutations=config.number_permutations)


    def forward(self,
                node_opcode: torch.Tensor, # (bs, num_nodes)
                node_feat: torch.Tensor, # (bs, num_nodes, num_node_feats)
                edges_adjecency: torch.Tensor, # (bs, num_nodes, num_nodes)
                node_attn_mask: torch.Tensor, # (bs, num_nodes)
                node_config_feat: torch.Tensor, # (bs, num_configs, num_config_nodes, num_node_feats)
                node_config_ids: torch.Tensor, # (bs, num_configs, num_config_nodes)
                config_idxs: Optional[torch.Tensor] = None, # (bs, num_configs)
                config_runtime: Optional[torch.Tensor] = None,):

        last_hidden_state = self.node_encoder(node_opcode,
                                    node_feat,
                                    edges_adjecency,
                                    node_attn_mask,
                                    node_config_feat,
                                    node_config_ids)

        output = self.head(last_hidden_state[:,:,0]).squeeze(-1)
        outputs = {'outputs': output, 'order': torch.argsort(output, dim=1)}
        if config_runtime is not None:
            loss = 0
            loss += self.loss_fn(output, config_runtime, config_idxs)
            outputs['loss'] = loss
        return outputs
