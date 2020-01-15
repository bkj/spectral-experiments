#!/usr/bin/env python

"""
    jhu_vn.py
"""

import json
import numpy as np
import pandas as pd
import networkx as nx
from tqdm import tqdm

from sklearn import metrics
from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split

from graspy.embed import AdjacencySpectralEmbed as ASE
from graspy.utils import pass_to_ranks

from rsub import *
from matplotlib import pyplot as plt

np.random.seed(1)

prop_training = 0.1

# --
# IO

graph_inpath = 'data/DS72784/subj1-scan1.graphml'
G = nx.read_graphml(graph_inpath)

labels = pd.read_csv('./data/DS72784/DS72784_desikan.csv')
labels = labels.set_index('dsreg')
label_lookup = dict(zip(labels.index, labels.values.argmax(axis=-1)))

node_names = [int(G.nodes[n]['name']) for n in G.nodes]
y          = np.array([label_lookup[n] for n in node_names])

# --
# Fit ASE

A     = nx.to_numpy_array(G)
A     = pass_to_ranks(A, method='simple-nonzero')
X_hat = ASE().fit_transform(A)

# --
# Write to disk

from scipy import sparse

row, col = np.where(A2)
val      = A2[(row, col)]
np.save('A2_sparse', np.column_stack([row, col, val]))
np.save('y', y)

# --
# Train model

from sklearn.ensemble import RandomForestClassifier

nX_hat = X_hat / np.sqrt((X_hat ** 2).sum(axis=-1, keepdims=True))

X_train, X_test, y_train, y_test = \
    train_test_split(nX_hat, y == 1, train_size=0.05, stratify=y)

model     = RandomForestClassifier(n_estimators=512, n_jobs=10).fit(X_train, y_train)
pred_test = model.predict_proba(X_test)[:,1]
metrics.roc_auc_score(y_test, pred_test)
