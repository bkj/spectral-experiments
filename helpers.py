#!/usr/bin/env python

"""
    helpers.py
"""

import random
import numpy as np
import scipy.sparse as sp

import torch

def set_seeds(seed):
    _ = random.seed(seed + 1)
    _ = np.random.seed(seed + 2)
    _ = torch.manual_seed(seed + 3)
    _ = torch.cuda.manual_seed(seed + 4)

def save_csr(path, x):
    row, col = np.where(x)
    val      = x[(row, col)]
    np.save(path, np.column_stack([row, col, val]))

def load_csr(path):
    row, col, val = np.load(path).T
    row, col = row.astype(np.int), col.astype(np.int)
    return sp.csr_matrix((val, (row, col)))

def train_stop_valid_split(n, p, random_state=None):
    assert len(p) == 3
    
    if random_state is not None:
        rng = np.random.RandomState(seed=random_state)
    else:
        rng = np.random
    
    folds = rng.choice(['train', 'stop', 'valid'], n_nodes, p=p)
    
    idx_train = np.where(folds == 'train')[0]
    idx_stop  = np.where(folds == 'stop')[0]
    idx_valid = np.where(folds == 'valid')[0]
    
    return idx_train, idx_stop, idx_valid

def to_numpy(x):
    return x.detach().cpu().numpy()