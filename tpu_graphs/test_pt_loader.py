import time
from collections import defaultdict

from torch.utils.data import DataLoader
from .pt_loader import *
from .score import speed_score

def test_loads_batch():
  random.seed(42)
  torch.manual_seed(0)
  filenames = get_files('tile', 'valid')

  dataset = LayoutDataset(filenames=filenames)
  sampler = BufferedRandomSampler(len(dataset))
  bs = 64
  dataloader = DataLoader(dataset, batch_size=bs, shuffle=False, collate_fn=custom_collate_fn, sampler=sampler)

  config_feat, node_feat, config_runtime, edge_idx, file_idx, trial_idx = next(iter(dataloader))

  assert config_feat.shape == (bs, 24)
  assert node_feat.shape == (bs, 27, 141)

  for i in range(30):
    config_feat, node_feat, config_runtime, edge_idx, file_idx, trial_idx = next(iter(dataloader))

  assert config_feat.shape == (bs, 24)
  assert node_feat.shape == (bs, 52, 141)

def test_pt_loader():
  random.seed(0)
  torch.manual_seed(0)
  filenames = get_files('tile', 'valid')

  dataset = LayoutDataset(filenames=filenames)
  sampler = BufferedRandomSampler(len(dataset))
  bs = 64
  dataloader = DataLoader(dataset, batch_size=bs, shuffle=False, collate_fn=custom_collate_fn, sampler=sampler)

  start = time.time()
  all_preds = defaultdict(list)
  def random_model(node_feat, config_feat, batch_size):
      return torch.rand(batch_size)


  for i, data in enumerate(dataloader):
      config_feat, node_feat, config_runtime, edge_idx, file_idx, trial_idx = data

      current_batch_size = trial_idx.shape[0]

      preds = random_model(node_feat, config_feat, current_batch_size)

      for j in range(current_batch_size):
        all_preds[file_idx[j].item()].append({'runtime': config_runtime[j].item(), 'pred': preds[j].item()})

      if i > 0 and  i % 300 == 0:
          break

  scores = []
  for file in all_preds.values():
      file = sorted(file, key=lambda x: x['runtime'])
      for i, f in enumerate(file):
          f['index'] = i

      file = sorted(file, key=lambda x: x['pred'])
      indices = np.array([f['index'] for f in file])
      runtimes = np.array([f['runtime'] for f in file])


      scores.append(speed_score(runtimes, indices, 3))

  assert np.mean(scores) == 0.00966974215854273

if __name__ == '__main__':
  test_pt_loader()
