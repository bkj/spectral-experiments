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
    if isinstance(x, np.ndarray):
        row, col = np.where(x)
        val      = x[(row, col)]
    elif isinstance(x, sp.csr_matrix):
        x_coo = x.tocoo()
        row, col, val = x_coo.row, x_coo.col, x_coo.data
    else:
        raise Exception()
    
    row = row.astype(np.float64)
    col = col.astype(np.float64)
    val = val.astype(np.float64)
    
    np.save(path, np.column_stack([row, col, val]))

def load_csr(path, square=False):
    row, col, val = np.load(path).T
    row, col = row.astype(np.int), col.astype(np.int)
    
    nrow = row.max() + 1
    ncol = col.max() + 1
    if square:
        nrow = ncol = max(nrow, ncol)
    
    return sp.csr_matrix((val, (row, col)), shape=(nrow, ncol))

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

def get_lcc(adj, y):
    _, comps = sp.csgraph.connected_components(adj)
    sel = comps == 0
    return adj[sel][:,sel], y[sel]