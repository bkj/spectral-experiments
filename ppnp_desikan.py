#!/usr/bin/env python

"""
    ppr_desikan.py
"""

import sys
import argparse
import numpy as np
from time import time
from tqdm import trange

from sklearn import metrics
from sklearn.preprocessing import normalize
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

import torch
from torch import nn
from torch.nn import functional as F

from helpers import set_seeds, load_csr, to_numpy

from ez_ppnp.models import EmbeddingPPNP
from ez_ppnp.ppr import exact_ppr_joblib, PrecomputedPPR
from ez_ppnp.trainer import train_unsupervised

# --
# CLI

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--graph-inpath',  type=str,   default='./data/DS72784/subj1-scan1.A_ptr.npy')
    parser.add_argument('--label-inpath',  type=str,   default='./data/DS72784/subj1-scan1.y.npy')
    parser.add_argument('--ppr-inpath',    type=str)
    parser.add_argument('--ppr-outpath',   type=str)
    parser.add_argument('--p-train',       type=float, default=0.1)
    
    parser.add_argument('--ppr-alpha',     type=float, default=0.9)
    parser.add_argument('--epochs',        type=int,   default=1000)
    parser.add_argument('--batch-size',    type=int,   default=2048)
    parser.add_argument('--hidden-dim',    type=int,   default=8)
    parser.add_argument('--lr',            type=int,   default=0.01)
    
    parser.add_argument('--seed',          type=int,   default=123)
    args = parser.parse_args()
    
    assert (args.ppr_inpath is None) or (args.ppr_outpath is None), 'cannot set ppr_inpath and ppr_outpath'
    if args.ppr_inpath is None:
        assert args.ppr_outpath is not None, 'if no ppr_inpath, must set ppr_outpath to cache ppr matrix'
    
    return args

args = parse_args()
set_seeds(args.seed)

# --
# IO

# ... could read from scratch, but save time by 
# reading from `pass_to_ranks` preprocessed version

adj = load_csr(args.graph_inpath)
y   = np.load(args.label_inpath)

n_nodes = adj.shape[0]

# --
# Precompute PPR

if args.ppr_inpath is None:
    print(f'ppnp_desikan.py: computing PPNP, caching to {args.ppr_outpath}', file=sys.stderr)
    
    ppr_array = exact_ppr_joblib(adj, alpha=args.ppr_alpha)
    np.fill_diagonal(ppr_array, 0)
    np.save(args.ppr_outpath, ppr_array)
else:
    print(f'ppnp_desikan.py: loading PPNP from {args.ppr_inpath}', file=sys.stderr)
    
    ppr_array = np.load(args.ppr_inpath)

# --
# Train embedding

model = EmbeddingPPNP(
    ppr        = PrecomputedPPR(ppr=ppr_array),
    n_nodes    = n_nodes,
    hidden_dim = args.hidden_dim,
)

model = model.cuda()
model = model.train()

loss_hist = train_unsupervised(model, lr=args.lr, epochs=args.epochs, batch_size=args.batch_size)

# --
# Compute X_hat

idx_chunks = np.array_split(np.arange(n_nodes), n_nodes // args.batch_size)
with torch.no_grad():
    X_hat = np.row_stack([to_numpy(model(idx_chunk)[1]) for idx_chunk in idx_chunks])

nX_hat = normalize(X_hat, axis=1, norm='l2')

# --
# Train/test split

idx_train, idx_valid = train_test_split(np.arange(n_nodes), train_size=args.p_train, test_size=1 - args.p_train)

nX_train, nX_valid = nX_hat[idx_train], nX_hat[idx_valid]
y_train, y_valid   = y[idx_train], y[idx_valid]

# --
# Train model
# !! TODO -- Should tune model

clf = RandomForestClassifier(n_estimators=512, n_jobs=10)
clf = clf.fit(nX_train, y_train)

pred_valid = clf.predict(nX_valid)

print({
    "acc"      : metrics.accuracy_score(y_valid, pred_valid),
    "f1_macro" : metrics.f1_score(y_valid, pred_valid, average='macro'),
    "f1_micro" : metrics.f1_score(y_valid, pred_valid, average='micro'),
})
