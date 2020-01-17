#!/usr/bin/env python

"""
    main.py
"""

import sys
import json
import argparse
import numpy as np
from time import time
from functools import partial

from sklearn import metrics
from sklearn.preprocessing import normalize
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

import torch
from torch import nn
from torch.nn import functional as F

from helpers import set_seeds, load_csr, to_numpy, get_lcc
from embedders import smart_ppr, embed_ppnp, embed_ppnp_supervised, embed_ppr_svd, embed_ase, embed_lse

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
    
    parser.add_argument('--no-normalize-features', action="store_true")
    
    parser.add_argument('--seed', type=int, default=123)
    args = parser.parse_args()
    
    if args.se_components is not None:
        assert args.se_components == args.hidden_dim
    
    return args

args = parse_args()
set_seeds(args.seed)

# # >>
# print('manual testing')
# args.inpath        = './data/DS72784/subj1-scan1'
# args.se_components = 8
# # <<

# --
# IO

adj = load_csr(args.inpath + '.adj.npy', square=True)

adj = ((adj + adj.T) > 0).astype(np.float32) # symmetrize + binarize

y   = np.load(args.inpath + '.y.npy')

adj, y = get_lcc(adj, y)

n_nodes = adj.shape[0]
n_edges = adj.nnz

# --
# Train/test split

idx_train, idx_valid = train_test_split(np.arange(n_nodes), train_size=args.p_train, test_size=1 - args.p_train)
y_train, y_valid     = y[idx_train], y[idx_valid]

# --
# Fit embeddings

X_hats = {}
meta   = {}

ppr_array = smart_ppr(adj, alpha=args.ppr_alpha)

emb_fns = {
    # "ppnp_supervised" : partial(embed_ppnp_supervised, ppr_array=ppr_array, y=y, idx_train=idx_train, hidden_dim=args.hidden_dim, lr=args.lr, epochs=args.epochs, batch_size=args.batch_size),
    # "ppnp"            : partial(embed_ppnp, ppr_array=ppr_array, hidden_dim=args.hidden_dim, lr=args.lr, epochs=args.epochs, batch_size=args.batch_size),
    # "ppr_full"        : partial(embed_ppr_svd, ppr_array=ppr_array, n_components=args.hidden_dim),
    "ppr_sparse"      : partial(embed_ppr_svd, ppr_array=ppr_array, n_components=args.hidden_dim, topk=args.pprsvd_topk),
    # "ase"             : partial(embed_ase, adj=adj, n_components=args.se_components),
    # "lse"             : partial(embed_lse, adj=adj, n_components=args.se_components),
}

for k, fn in emb_fns.items():
    print(f'main: embedding {k}', file=sys.stderr)
    t         = time()
    X_hats[k] = fn()
    
    meta[k] = {
        "dim"     : X_hats[k].shape[1],
        "elapsed" : time() - t
    }
    print(f'\telapsed={meta[k]["elapsed"]}', file=sys.stderr)

print('-' * 10, file=sys.stderr)

# --
# Train model

def fit_predict(X_train, y_train, X_valid, normalize_X=True):
    if normalize_X:
        nX_train = normalize(X_train, axis=1, norm='l2')
        nX_valid = normalize(X_valid, axis=1, norm='l2')
    
    clf = RandomForestClassifier(n_estimators=512, n_jobs=10) # ?? Should use different classifier?
    clf = clf.fit(nX_train, y_train)
    
    return clf.predict(nX_valid)

def compute_metrics(act, pred):
    return  {
        "accuracy" : float(metrics.accuracy_score(act, pred)),
        "f1_macro" : float(metrics.f1_score(act, pred, average='macro')),
        "f1_micro" : float(metrics.f1_score(act, pred, average='micro')),
    }


no_model = set(['ppnp_supervised']) # don't train a model here

scores = {}
for k, X_hat in X_hats.items():
    print(f'main: modeling {k}', file=sys.stderr)
    
    if k in no_model:
        pred_valid = X_hat[idx_valid].argmax(axis=-1)
    else:
        pred_valid = fit_predict(X_train=X_hat[idx_train], y_train=y_train, X_valid=X_hat[idx_valid])
    
    scores[k]  = compute_metrics(act=y_valid, pred=pred_valid)

# --
# Log

print(json.dumps({
    "dataset" : str(args.inpath),
    "n_nodes" : int(n_nodes),
    "n_edges" : int(n_edges),
    "meta"    : meta,
    "scores"  : scores,
}))
