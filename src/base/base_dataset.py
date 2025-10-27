
import numpy as np
import torch
from torch.utils.data import Dataset

class DictDataset(Dataset):
    def __init__(self, X, y, g=None):
        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y, dtype=np.int64)
        if g is None:
            g = np.zeros_like(y)
        else:
            g = np.asarray(g, dtype=np.int64)
        self.X = torch.from_numpy(X)
        self.y = torch.from_numpy(y)
        self.g = torch.from_numpy(g)
    def __len__(self):
        return self.X.shape[0]
    def __getitem__(self, idx):
        return {'x': self.X[idx], 'y': self.y[idx], 'g': self.g[idx]}

class BaseADDataset:
    def __init__(self, train_set, val_set=None, test_set=None):
        self.train_set = train_set
        self.val_set = val_set if val_set is not None else train_set
        self.test_set = test_set if test_set is not None else val_set if val_set is not None else train_set
