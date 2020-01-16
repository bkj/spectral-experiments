#!/usr/bin/env python

"""
    ppr_desikan.py
"""

import sys
import json
import argparse
import numpy as np
from time import time
from tqdm import trange
from scipy.sparse import csgraph

from sklearn import metrics
from sklearn.preprocessing import normalize
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

import torch
from torch import nn
from torch.nn import functional as F

from helpers import set_seeds, load_csr, to_numpy, get_lcc

from embedders import embed_ppnp, embed_ase, embed_lse

# --
# CLI

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--inpath',  type=str,   default='./data/cora/cora')
    
    parser.add_argument('--ppr-inpath',    type=str)
    parser.add_argument('--ppr-outpath',   type=str,   default='delete-me')
    parser.add_argument('--p-train',       type=float, default=0.1)
    
    parser.add_argument('--ppr-alpha',     type=float, default=0.1)
    parser.add_argument('--epochs',        type=int,   default=500)
    parser.add_argument('--batch-size',    type=int,   default=2048)
    parser.add_argument('--hidden-dim',    type=int,   default=8)
    parser.add_argument('--lr',            type=int,   default=0.01)
    
    parser.add_argument('--active-permute',        action="store_true")
    parser.add_argument('--no-normalize-features', action="store_true")
    
    parser.add_argument('--seed', type=int, default=123)
    args = parser.parse_args()
    
    assert (args.ppr_inpath is None) or (args.ppr_outpath is None), 'cannot set ppr_inpath and ppr_outpath'
    if args.ppr_inpath is None:
        assert args.ppr_outpath is not None, 'if no ppr_inpath, must set ppr_outpath to cache ppr matrix'
    
    return args

args = parse_args()
set_seeds(args.seed)

# --
# IO

adj = load_csr(args.inpath + '.adj.npy', square=True)
adj = ((adj + adj.T) > 0).astype(np.float32)
y   = np.load(args.inpath + '.y.npy')

adj, y = get_lcc(adj, y)

n_nodes = adj.shape[0]
n_edges = adj.nnz

if args.active_permute:
    p   = np.random.permutation(n_nodes)
    adj = adj[p][:,p]
    y   = y[p]

# --
# Fit embeddings

# n_components = args.hidden_dim # fix ase/lse dimension
n_components = None              # automatic selection

print('embed_ppnp', file=sys.stderr)
X_ppnp = embed_ppnp(adj, args.ppr_alpha, args.hidden_dim, args.lr, args.epochs, args.batch_size)

print('embed_ase', file=sys.stderr)
X_ase  = embed_ase(adj, n_components=n_components)

print('embed_lse', file=sys.stderr)
X_lse  = embed_lse(adj, n_components=n_components)

if not args.no_normalize_features:
    X_ppnp = normalize(X_ppnp, axis=1, norm='l2')
    X_ase  = normalize(X_ase, axis=1, norm='l2')
    X_lse  = normalize(X_lse, axis=1, norm='l2')

# --
# Train/test split

idx_train, idx_valid = train_test_split(np.arange(n_nodes), train_size=args.p_train, test_size=1 - args.p_train)

X_ppnp_train, X_ppnp_valid = X_ppnp[idx_train], X_ppnp[idx_valid]
X_ase_train, X_ase_valid   = X_ase[idx_train], X_ase[idx_valid]
X_lse_train, X_lse_valid   = X_lse[idx_train], X_lse[idx_valid]
y_train, y_valid           = y[idx_train], y[idx_valid]

# --
# Train model

def do_score(X_train, y_train, X_valid):
    clf = RandomForestClassifier(n_estimators=512, n_jobs=10)
    clf = clf.fit(X_train, y_train)
    pred_valid = clf.predict(X_valid)
    return  {
        "accuracy" : float(metrics.accuracy_score(y_valid, pred_valid)),
        "f1_macro" : float(metrics.f1_score(y_valid, pred_valid, average='macro')),
        "f1_micro" : float(metrics.f1_score(y_valid, pred_valid, average='micro')),
    }

ppnp_scores = do_score(X_ppnp_train, y_train, X_ppnp_valid)
ase_scores  = do_score(X_ase_train, y_train, X_ase_valid)
lse_scores  = do_score(X_lse_train, y_train, X_lse_valid)

print(json.dumps({
    "_metadata" : {
        "dataset" : str(args.inpath),
        "n_nodes" : int(n_nodes),
        "n_edges" : int(n_edges),
        
        "ppnp_dim" : X_ppnp.shape[1],
        "ase_dim"  : X_ase.shape[1],
        "lse_dim"  : X_lse.shape[1],
    },
    "ppnp_scores" : ppnp_scores,
    "ase_scores"  : ase_scores,
    "lse_scores"  : lse_scores,
}))
