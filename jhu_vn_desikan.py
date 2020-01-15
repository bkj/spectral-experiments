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

from helpers import set_seeds, load_csr

# --
# CLI

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--graph-inpath',  type=str,   default='./data/DS72784/subj1-scan1.A_ptr.npy')
    parser.add_argument('--label-inpath',  type=str,   default='./data/DS72784/subj1-scan1.y.npy')
    parser.add_argument('--p-train',       type=float, default=0.1)
    parser.add_argument('--seed',          type=int,   default=123)
    return parser.parse_args()


args = parse_args()
set_seeds(args.seed)

# --
# IO

A_ptr = load_csr(args.graph_inpath).toarray()
y     = np.load(args.label_inpath)

n_nodes = A_ptr.shape[0]

# --
# Compute ASE

X_hat  = AdjacencySpectralEmbed().fit_transform(A_ptr)
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
