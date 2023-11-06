# import pytest
# from .pt_loader import BufferedRandomSampler
# import numpy as np

# def test_buffered_random_sampler():
#     # Create an instance of BufferedRandomSampler
#     sampler = BufferedRandomSampler(10)

#     # Get a set of indices
#     indices1 = sampler.get_indices()

#     # Get another set of indices
#     indices2 = sampler.get_indices()

#     # Check if the two sets of indices are not the same, implying randomness
#     assert not np.array_equal(indices1, indices2)


import torch


t = torch.zeros(64, 128, 69)

print(t.flatten(1, 2).shape)
