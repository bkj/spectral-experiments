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

# --
# Helpers

def save_csr(path, x):
    row, col = np.where(x)
    val      = x[(row, col)]
    np.save(path, np.column_stack([row, col, val]))

def load_csr(path):
    row, col, val = np.load(path).T
    row, col = row.astype(np.int), col.astype(np.int)
    return sp.csr_matrix((val, (row, col)))

# --
# CLI

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--graph-inpath',  type=str, default='./data/DS72784/subj1-scan1.graphml')
    parser.add_argument('--label-inpath',  type=str, default='./data/DS72784/DS72784_desikan.csv')
    parser.add_argument('--p-train',       type=float, default=0.1)
    # parser.add_argument('--n-iters',       type=int,   default=32)
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

# --
# Fit ASE

A     = nx.to_numpy_array(G)
A_ptr = pass_to_ranks(A, method='simple-nonzero')
save_csr('data/DS72784/subj1-scan1.A_ptr.npy', A_ptr)
np.save('data/DS72784/subj1-scan1.y.npy', y)

X_hat  = AdjacencySpectralEmbed().fit_transform(A_ptr)
nX_hat = normalize(X_hat, axis=1, norm='l2')

# --
# Train model

X_train, X_test, y_train, y_test = \
    train_test_split(nX_hat, y == 1, train_size=0.05, stratify=y == 1)

model     = RandomForestClassifier(n_estimators=512, n_jobs=10).fit(X_train, y_train)
pred_test = model.predict_proba(X_test)[:,1]
metrics.roc_auc_score(y_test, pred_test)
