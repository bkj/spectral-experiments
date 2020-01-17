#!/usr/bin/env python

"""
    main.py
"""

import sys
import json
import argparse
import numpy as np
from functools import partial

from sklearn import metrics
from sklearn.preprocessing import normalize
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

import torch
from torch import nn
from torch.nn import functional as F

from helpers import set_seeds, load_csr, to_numpy, get_lcc
from embedders import embed_ppnp, embed_ppr_svd, embed_ase, embed_lse

# --
# CLI

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--inpath',  type=str,   default='./small_data/cora/cora')
    parser.add_argument('--p-train', type=float, default=0.1)
    
    parser.add_argument('--ppr-alpha',     type=float, default=0.1)
    
    # ppnp
    parser.add_argument('--epochs',        type=int,   default=500)
    parser.add_argument('--batch-size',    type=int,   default=2048)
    parser.add_argument('--hidden-dim',    type=int,   default=8)
    parser.add_argument('--lr',            type=int,   default=0.01)
    
    # ppr_svd
    parser.add_argument('--pprsvd-topk',   type=int,   default=128)
    
    # ase/lse
    parser.add_argument('--se-components', type=int,   default=None)
    
    parser.add_argument('--active-permute',        action="store_true")
    parser.add_argument('--no-normalize-features', action="store_true")
    
    parser.add_argument('--seed', type=int, default=123)
    args = parser.parse_args()
    
    if args.se_components is not None:
        assert args.se_components == args.hidden_dim
    
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
    # !! (pro)actively permute data, in case order leaks information
    p   = np.random.permutation(n_nodes)
    adj = adj[p][:,p]
    y   = y[p]

# --
# Fit embeddings

X_hats = {}

emb_fns = {
    # "ppnp" : partial(embed_ppnp, ppr_alpha=args.ppr_alpha, hidden_dim=args.hidden_dim, lr=args.lr, epochs=args.epochs, batch_size=args.batch_size)
    "ppr_full"   : partial(embed_ppr_svd, ppr_alpha=args.ppr_alpha, n_components=args.hidden_dim),
    "ppr_sparse" : partial(embed_ppr_svd, ppr_alpha=args.ppr_alpha, n_components=args.hidden_dim, topk=args.pprsvd_topk),
    "ase"        : partial(embed_ase, n_components=args.se_components),
    "lsa"        : partial(embed_lse, n_components=args.se_components),
}

for k, fn in emb_fns.items():
    print(f'main: embedding {k}', file=sys.stderr)
    X_hats[k] = fn(adj)

# --
# Train/test split

idx_train, idx_valid = train_test_split(np.arange(n_nodes), train_size=args.p_train, test_size=1 - args.p_train)
y_train, y_valid     = y[idx_train], y[idx_valid]

# --
# Train model

def do_score(X_train, y_train, X_valid):
    nX_train = normalize(X_train, axis=1, norm='l2')
    nX_valid = normalize(X_valid, axis=1, norm='l2')
    
    clf = RandomForestClassifier(n_estimators=512, n_jobs=10)
    clf = clf.fit(nX_train, y_train)
    
    pred_valid = clf.predict(nX_valid)
    return  {
        "accuracy" : float(metrics.accuracy_score(y_valid, pred_valid)),
        "f1_macro" : float(metrics.f1_score(y_valid, pred_valid, average='macro')),
        "f1_micro" : float(metrics.f1_score(y_valid, pred_valid, average='micro')),
    }

scores = {}
for k, X_hat in X_hats.items():
    print(f'main: modeling {k}', file=sys.stderr)
    scores[k] = do_score(X_train=X_hat[idx_train], y_train=y_train, X_valid=X_hat[idx_valid])

# --
# Log

print(json.dumps({
    "_metadata" : {
        "dataset" : str(args.inpath),
        "n_nodes" : int(n_nodes),
        "n_edges" : int(n_edges),
    },
    "scores" : scores
}))
