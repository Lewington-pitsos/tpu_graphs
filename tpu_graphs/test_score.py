import numpy as np
from .score import speed_score
from .pt_loader import get_files

def test_scores():
    np.random.seed(42)
    val_tile_files = get_files('tile', 'valid')

    all_scores = []
    rand_scores = []

    for filename in val_tile_files:
        unnormalised_runtime = np.load(filename, allow_pickle=True)['config_runtime']
        runtime_normalisers = np.load(filename, allow_pickle=True)['config_runtime_normalizers']
        runtimes = unnormalised_runtime / runtime_normalisers

        perfect_preds = np.argsort(runtimes)
        perfect_score = speed_score(runtimes, perfect_preds, 3)
        np.random.shuffle(perfect_preds)
        random_score = speed_score(runtimes, perfect_preds, 3)

        rand_scores.append(random_score)
        all_scores.append(perfect_score)

    assert np.mean(all_scores) == 1.0
    assert np.mean(rand_scores) == 0.10694134643762006
