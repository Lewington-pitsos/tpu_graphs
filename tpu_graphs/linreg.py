import numpy as np
from .score import speed_score
from .pt_loader import get_files
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error


def load_data(files, idx):
    X, y = [], []

    for filename in files:
        unnormalised_runtime = np.load(filename, allow_pickle=True)['config_runtime']
        runtime_normalisers = np.load(filename, allow_pickle=True)['config_runtime_normalizers']
        runtimes = unnormalised_runtime / runtime_normalisers

        output_bounds_sums = np.load(filename, allow_pickle=True)['config_feat'][:, idx]

        X.extend(output_bounds_sums)
        y.extend(runtimes)

    return np.array(X), np.array(y).reshape(-1, 1)

def train():
    idx = [6,7, 14, 15]
    np.random.seed(42)
    train_file_files = get_files('tile', 'train')
    X, y = load_data(train_file_files, idx)

    print('X shape', X.shape)
    print('y shape', y.shape)

    model = LinearRegression()

    model.fit(X, y)
    print("Model coefficients:", model.coef_)
    print("Model intercept:", model.intercept_)

    train_mse = mean_squared_error(y, model.predict(X))
    print("Train MSE:", train_mse)

    val_file_files = get_files('tile', 'valid')
    X_val, y_val = load_data(val_file_files, idx)
    y_pred = model.predict(X_val)
    mse = mean_squared_error(y_val, y_pred)

    print("Mean Squared Error:", mse)

    default_mse = mean_squared_error(y_val, np.mean(y_val) * np.ones_like(y_val))

    print("default MSE", default_mse)

    all_scores = []
    rand_scores = []

    for filename in val_file_files:
        unnormalised_runtime = np.load(filename, allow_pickle=True)['config_runtime']
        runtime_normalisers = np.load(filename, allow_pickle=True)['config_runtime_normalizers']
        runtimes = unnormalised_runtime / runtime_normalisers

        perfect_preds = np.argsort(runtimes)
        perfect_score = speed_score(runtimes, perfect_preds, 3)
        predicted_runtimes = model.predict(np.load(filename, allow_pickle=True)['config_feat'][:, idx])
        other_preds = np.argsort(predicted_runtimes[:, 0])
        linear_score = speed_score(runtimes, other_preds, 3)

        rand_scores.append(linear_score)
        all_scores.append(perfect_score)

    print("perfect score", np.mean(all_scores))
    print("linear score", np.mean(rand_scores))

train()
