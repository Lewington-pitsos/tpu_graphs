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
        self.current_file_idx = -1
        self.trials_per_file = self.precompute_trials_per_file()
        self.cumulative_trials = np.cumsum(self.trials_per_file)

    def precompute_trials_per_file(self):
        trials_per_file = []
        for filename in self.filenames:
            with np.load(filename, allow_pickle=True) as data:
                trials_per_file.append(len(data['config_runtime']))
        return trials_per_file

    def load_file(self, filename):
        if self.current_file_data is not None:
            del self.current_file_data  # Close current file data if any
        self.current_file_data = dict(np.load(filename, allow_pickle=True))

    def __len__(self):
        return self.cumulative_trials[-1]

    def __getitem__(self, idx):
        if idx >= self.__len__():
            raise IndexError("Index out of range")

        file_idx = np.searchsorted(self.cumulative_trials, idx, side='right')
        if file_idx != self.current_file_idx:
            self.load_file(self.filenames[file_idx])
            self.current_file_idx = file_idx

        trial_idx = idx - self.cumulative_trials[file_idx - 1] if file_idx > 0 else idx
        return self.get_trial_data(self.current_file_data, file_idx, trial_idx)

    def get_trial_data(self, file_data, file_idx, trial_idx):
        config_feat = torch.from_numpy(file_data['config_feat'][trial_idx])
        node_feat = torch.from_numpy(file_data['node_feat'])
        node_opcode = torch.from_numpy(file_data['node_opcode'])
        config_runtime = torch.tensor([file_data['config_runtime'][trial_idx] / file_data['config_runtime_normalizers'][trial_idx]])

        node_feat = torch.concat([node_feat, node_opcode.unsqueeze(1)], axis=1)

        return config_feat, node_feat, config_runtime, torch.tensor([file_idx]), torch.tensor([trial_idx])



def pad_sequence(sequences, batch_first=True, padding_value=-1):
    max_len = max([s.size(0) for s in sequences])
    batch_size = len(sequences)
    max_size = sequences[0].size(1)
    padded_batch = torch.full((batch_size, max_len, max_size), padding_value)
    for i, sequence in enumerate(sequences):
        length = sequence.size(0)
        padded_batch[i, :length] = sequence
    return padded_batch

def custom_collate_fn(batch):
    config_feat_list, node_feat_list, config_runtime_list, file_idx, trial_idx = zip(*batch)

    config_feat = torch.stack(config_feat_list)
    config_runtime = torch.stack(config_runtime_list)
    file_idxs = torch.stack(file_idx)
    trial_idxs = torch.stack(trial_idx)



    node_feat_padded = pad_sequence(node_feat_list, batch_first=True)

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
