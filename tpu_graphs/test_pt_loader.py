import time
from collections import defaultdict

from torch.utils.data import DataLoader
from .pt_loader import *
from .score import speed_score

filenames = get_files('tile', 'valid')


dataset = LayoutDataset(filenames=filenames)
sampler = BufferedRandomSampler(len(dataset))
dataloader = DataLoader(dataset, batch_size=64, shuffle=False, collate_fn=custom_collate_fn)


start = time.time()
preds = defaultdict(list)
for i, data in enumerate(dataloader):
    # if i > 5000:
    #     break
    pass

print(time.time() - start)
