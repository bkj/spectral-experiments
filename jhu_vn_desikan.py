#!/usr/bin/env python

"""
    jhu_vn_desikan.py
"""

import argparse
import numpy as np
import pandas as pd
import networkx as nx
from scipy import sparse as sp

from sklearn import metrics
from sklearn.preprocessing import normalize
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

from graspy.utils import pass_to_ranks
from graspy.embed import AdjacencySpectralEmbed

from helpers import save_csr, train_stop_valid_split

# --
# CLI

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--graph-inpath',  type=str, default='./data/DS72784/subj1-scan1.graphml')
    parser.add_argument('--label-inpath',  type=str, default='./data/DS72784/DS72784_desikan.csv')
    parser.add_argument('--p-train',       type=float, default=0.1)
    parser.add_argument('--seed',          type=int,   default=123)
    return parser.parse_args()


args = parse_args()
np.random.seed(args.seed)

# --
# IO

G = nx.read_graphml(args.graph_inpath)

labels       = pd.read_csv(args.label_inpath).set_index('dsreg')
label_lookup = dict(zip(labels.index, labels.values.argmax(axis=-1)))

node_names  = [int(G.nodes[n]['name']) for n in G.nodes]
y           = np.array([label_lookup[n] for n in node_names])

# Remap y to sequential zero-indexed integers
uy      = list(set(y))
y_remap = dict(zip(uy, range(len(uy))))
y       = np.array([y_remap[yy] for yy in y])

n_nodes = len(G.nodes)

# --
# Fit ASE

A      = nx.to_numpy_array(G)
A_ptr  = pass_to_ranks(A, method='simple-nonzero')
X_hat  = AdjacencySpectralEmbed().fit_transform(A_ptr)
nX_hat = normalize(X_hat, axis=1, norm='l2')

save_csr('data/DS72784/subj1-scan1.A_ptr.npy', A_ptr)
np.save('data/DS72784/subj1-scan1.y.npy', y)

# --
# Train/test split
# `train_stop_valid_split` is used to match `ppnp` methods, which
#  may use a `stop` split for early stopping

idx_train, idx_stop, idx_valid = \
    train_stop_valid_split(n_nodes, p=[0.05, 0.05, 0.9], random_state=111)

idx_train = np.concatenate([idx_train, idx_stop])
del idx_stop

nX_train, nX_valid = nX_hat[idx_train], nX_hat[idx_valid]
y_train, y_valid   = y[idx_train], y[idx_valid]

# --
# Train model
# !! TODO -- Should tune model

clf = RandomForestClassifier(n_estimators=512, n_jobs=10)
clf = clf.fit(nX_train, y_train)

prob_valid = clf.predict_proba(nX_valid)
pred_valid = prob_valid.argmax(axis=-1)

print({
    "acc"      : metrics.accuracy_score(y_valid, pred_valid),
    "f1_macro" : metrics.f1_score(y_valid, pred_valid, average='macro'),
    "f1_micro" : metrics.f1_score(y_valid, pred_valid, average='micro'),
})



