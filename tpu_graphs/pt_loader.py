import os
import random
import numpy as np
import torch
from torch.utils.data import Dataset
from .constants import *

def get_files(collection, split):
    if collection == 'tile':
        if split == 'train':
            return [os.path.join(TRAIN_DIR, filename) for filename in os.listdir(TRAIN_DIR)]
        elif split == 'valid':
            return [os.path.join(VALID_DIR, filename) for filename in os.listdir(VALID_DIR)]
        elif split == 'test':
            return [os.path.join(TEST_DIR, filename) for filename in os.listdir(TEST_DIR)]
        else:
            raise ValueError("Invalid split")
    raise ValueError("Invalid collection")

class LayoutDataset(Dataset):
    def __init__(self, filenames):
        self.filenames = filenames
        self.current_file_data = None
        self.trials_per_file = self.precompute_trials_per_file()
        self.cumulative_trials = np.cumsum(self.trials_per_file)
        self.open_file_indices = []
        self.open_files = []

    def precompute_trials_per_file(self):
        trials_per_file = []
        for filename in self.filenames:
            with np.load(filename, allow_pickle=True) as data:
                trials_per_file.append(len(data['config_runtime']))
        return trials_per_file

    def load_new_file(self, file_idx):
        if len(self.open_file_indices) >= 50:
          self.open_file_indices.pop(0)
          self.open_files.pop(0)

        filename = self.filenames[file_idx]

        self.current_file_data = dict(np.load(filename, allow_pickle=True))
        self.open_file_indices.append(file_idx)
        self.open_files.append(self.current_file_data)

        self.current_file_data = self.open_files[-1]


    def __len__(self):
        return self.cumulative_trials[-1]

    def __getitem__(self, idx):
        if idx >= self.__len__():
            raise IndexError("Index out of range")

        file_idx = np.searchsorted(self.cumulative_trials, idx, side='right')
        if file_idx not in self.open_file_indices:
            self.load_new_file(file_idx)
        else:
            self.current_file_data = self.open_files[self.open_file_indices.index(file_idx)]

        trial_idx = idx - self.cumulative_trials[file_idx - 1] if file_idx > 0 else idx
        return self.get_trial_data(self.current_file_data, file_idx, trial_idx)

    def get_trial_data(self, file_data, file_idx, trial_idx):
        config_feat = file_data['config_feat'][trial_idx]
        node_feat = file_data['node_feat']
        node_opcode = file_data['node_opcode']
        config_runtime = [file_data['config_runtime'][trial_idx] / file_data['config_runtime_normalizers'][trial_idx]]

        node_feat = np.concatenate([node_feat, node_opcode.reshape(-1, 1)], axis=1)

        return config_feat, node_feat, config_runtime, file_idx, trial_idx

def pad_sequence(sequences, padding_value=-1):
    # Convert the list of numpy arrays to a list of tensors
    sequences = [torch.tensor(s) for s in sequences]

    # Use the PyTorch pad_sequence function to pad all tensors to the maximum length
    padded_batch = torch.nn.utils.rnn.pad_sequence(sequences, batch_first=True, padding_value=padding_value)

    return padded_batch

# Example usage:
# sequences = [np.random.randn(10, 5), np.random.randn(8, 5), np.random.randn(12, 5)]
# padded_sequences = pad_sequence_efficient(sequences)


def custom_collate_fn(batch):
    config_feat_list, node_feat_list, config_runtime_list, file_idx_list, trial_idx_list = zip(*batch)

    config_feat = torch.tensor(np.array(config_feat_list))
    config_runtime = torch.tensor(config_runtime_list)
    file_idxs = torch.tensor(file_idx_list)
    trial_idxs = torch.tensor(trial_idx_list)



    node_feat_padded = pad_sequence(node_feat_list)

    return config_feat, node_feat_padded, config_runtime, file_idxs, trial_idxs

class BufferedRandomSampler:
    def __init__(self, data_source_length, buffer_size=200):
        self.data_source_length = data_source_length
        self.buffer_size = buffer_size
        self.buffer = []
        self.index_iter = iter(range(data_source_length))

    def fill_buffer(self):
        try:
            while len(self.buffer) < self.buffer_size:
                self.buffer.append(next(self.index_iter))
        except StopIteration:
            pass

    def __iter__(self):
        return self

    def __next__(self):
        if not self.buffer:
            self.fill_buffer()
            if not self.buffer:
                raise StopIteration

        random.shuffle(self.buffer)
        return self.buffer.pop()
