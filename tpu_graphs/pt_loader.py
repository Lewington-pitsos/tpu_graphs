import os
import random
import numpy as np
import torch
from torch.utils.data import Dataset

TILE_DIR = 'data/npz_all/npz/tile/xla/'
TRAIN_DIR = TILE_DIR + 'train/'
VALID_DIR = TILE_DIR + 'valid/'
TEST_DIR = TILE_DIR + 'test/'


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
        if len(self.open_file_indices) >= 150:
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


# tile valid
# Config Features Mean: 17.3280029296875, Standard Deviation: 82.76725006103516
# Node Features Mean: 19.088706970214844, Standard Deviation: 368.8953552246094
# Config Runtime Mean: 5.287334825787326, Standard Deviation: 8.122184066667144

# tile train
# Config Features Mean: 16.741966247558594, Standard Deviation: 74.34544372558594
# Node Features Mean: 14.231035232543945, Standard Deviation: 305.2548828125
# Config Runtime Mean: 4.980834250716549, Standard Deviation: 8.203627220003426

def layout_normalize(
        config_feat,
        node_feat,
        config_runtime,
        config_feat_mean=16.741966247558594,
        config_feat_std=74.34544372558594,
        node_feat_mean=14.231035232543945,
        node_feat_std=305.2548828125,
        config_runtime_mean=4.980834250716549,
        config_runtime_std=8.203627220003426
    ):

    mask = node_feat >= 0
    node_feat  = torch.sqrt(node_feat * mask)

    config_feat = (config_feat - config_feat_mean) / config_feat_std
    node_feat = (node_feat - node_feat_mean) / node_feat_std
    config_runtime = (config_runtime - config_runtime_mean) / config_runtime_std
    return config_feat, node_feat, config_runtime

def pad_sequence(sequences, padding_value=0):
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

    config_feat, node_feat_padded, config_runtime = layout_normalize(config_feat, node_feat_padded, config_runtime)

    return config_feat, node_feat_padded, config_runtime, file_idxs, trial_idxs

class BufferedRandomSampler:
    def __init__(self, data_source_length, buffer_size=200):
        self.data_source_length = data_source_length
        self.buffer_size = buffer_size
        self.buffer = []
        # Start with a fresh iterator
        self.index_iter = self._get_new_iterator()

    def _get_new_iterator(self):
        # Create a new iterator over the data source length
        return iter(range(self.data_source_length))

    def fill_buffer(self):
        try:
            while len(self.buffer) < self.buffer_size:
                self.buffer.append(next(self.index_iter))
        except StopIteration:
            # If StopIteration is called, it means the epoch ended
            # Reset the iterator and buffer for the next epoch
            self.index_iter = self._get_new_iterator()
            self.buffer = []

    def __iter__(self):
        return self

    def __next__(self):
        if not self.buffer:
            self.fill_buffer()
            random.shuffle(self.buffer)
            if not self.buffer:
                raise StopIteration
        return self.buffer.pop()

    def __len__(self):
        # If you want to use len(sampler), it should return the number of samples
        return self.data_source_length
