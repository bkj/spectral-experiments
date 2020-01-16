#!/usr/bin/env python

"""
    jhu_vn_desikan.py
"""

import sys
import json
import argparse
import numpy as np
import pandas as pd
import networkx as nx
from scipy import sparse as sp

from sklearn import metrics
from sklearn.preprocessing import normalize
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

from graspy.embed import AdjacencySpectralEmbed

from helpers import set_seeds, load_csr, get_lcc

# --
# CLI

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--inpath',  type=str,   default='./data/cora/cora')
    parser.add_argument('--p-train', type=float, default=0.1)
    parser.add_argument('--seed',    type=int,   default=123)
    return parser.parse_args()


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
print(json.dumps({
    "n_nodes" : n_nodes,
    "n_edges" : n_edges,
}), file=sys.stderr)

# --
# Compute ASE

print('jhu_vn_desikan.py: compute ASE', file=sys.stderr)

X_hat  = AdjacencySpectralEmbed().fit_transform(adj.toarray())
if isinstance(X_hat, tuple):
    X_hat = np.column_stack(X_hat)

nX_hat = normalize(X_hat, axis=1, norm='l2')

# --
# Train/test split

idx_train, idx_valid = train_test_split(np.arange(n_nodes), train_size=args.p_train, test_size=1 - args.p_train)

nX_train, nX_valid = nX_hat[idx_train], nX_hat[idx_valid]
y_train, y_valid   = y[idx_train], y[idx_valid]

# --
# Train model

clf = RandomForestClassifier(n_estimators=512, n_jobs=10)
clf = clf.fit(nX_train, y_train)

pred_valid = clf.predict(nX_valid)

print(json.dumps({
    "dataset"  : args.inpath,
    "method"   : "ase",
    "acc"      : float(metrics.accuracy_score(y_valid, pred_valid)),
    "f1_macro" : float(metrics.f1_score(y_valid, pred_valid, average='macro')),
    "f1_micro" : float(metrics.f1_score(y_valid, pred_valid, average='micro')),
}))
