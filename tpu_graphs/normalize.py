import torch
import random
from torch.utils.data import DataLoader

from .pt_loader import *

random.seed(42)
torch.manual_seed(0)

filenames = get_files('tile', 'valid')
dataset = LayoutDataset(filenames=filenames)
sampler = BufferedRandomSampler(len(dataset))
bs = 64
dataloader = DataLoader(dataset, batch_size=bs, shuffle=False, collate_fn=custom_collate_fn, sampler=sampler)


# Step 1: Initialize accumulators
config_feat_sum = 0
config_feat_sq_sum = 0
node_feat_sum = 0
node_feat_sq_sum = 0
config_runtime_sum = 0
config_runtime_sq_sum = 0
total_samples = 0

# Step 2: Iterate over the entire dataset
for config_feat, node_feat, config_runtime, _, _ in dataloader:

    if torch.isnan(config_feat).any() or torch.isinf(config_feat).any():
        print("config_feat contains NaN or Inf.")
        continue
    if torch.isnan(node_feat).any() or torch.isinf(node_feat).any():
        print("node_feat contains NaN or Inf.")
        continue
    if torch.isnan(config_runtime).any() or torch.isinf(config_runtime).any():
        print("config_runtime contains NaN or Inf.")
        continue

    config_feat_sum += config_feat.sum()
    config_feat_sq_sum += (config_feat ** 2).sum()
    node_feat_sum += node_feat.sum()
    node_feat_sq_sum += (node_feat ** 2).sum()
    config_runtime_sum += config_runtime.sum()
    config_runtime_sq_sum += (config_runtime ** 2).sum()
    total_samples += config_feat.shape[0]

# Step 4: Calculate mean and standard deviation
config_feat_mean = config_feat_sum / total_samples
config_feat_std = torch.sqrt(config_feat_sq_sum / total_samples - config_feat_mean ** 2)
node_feat_mean = node_feat_sum / total_samples
node_feat_std = torch.sqrt(node_feat_sq_sum / total_samples - node_feat_mean ** 2)
config_runtime_mean = config_runtime_sum / total_samples
config_runtime_std = torch.sqrt(config_runtime_sq_sum / total_samples - config_runtime_mean ** 2)

# Step 5: Print out the calculated mean and standard deviation
print(f"config_feat mean: {config_feat_mean}, std: {config_feat_std}")
print(f"node_feat mean: {node_feat_mean}, std: {node_feat_std}")
print(f"config_runtime mean: {config_runtime_mean}, std: {config_runtime_std}")
