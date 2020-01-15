#!/usr/bin/env python

"""
    jhu_vn.py
"""

import argparse
import numpy as np
import networkx as nx
from tqdm import tqdm

from sklearn import metrics
from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split

from graspy.embed import AdjacencySpectralEmbed, LaplacianSpectralEmbed

# --
# Helpers

ulabels = [
    "KC",
    "PN",
    "MBIN",
    "MBON",
]

n_class = len(ulabels)

def to_coarse_label(x):
    for l in ulabels:
        if l in x:
            return l
    
    return None

# --
# CLI

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--inpath',  type=str,   default='data/maggot/mb_2019-09-23/G.graphml')
    parser.add_argument('--p-train', type=float, default=0.1)
    parser.add_argument('--n-iters', type=int,   default=32)
    parser.add_argument('--mode',    type=str,   default='ase', choices=['ase', 'lse'])
    parser.add_argument('--seed',    type=int,   default=123)
    return parser.parse_args()

args = parse_args()
np.random.seed(args.seed)

# --
# IO

G = nx.read_graphml(args.inpath)
y = [to_coarse_label(xx) for xx in nx.get_node_attributes(G, 'Class').values()]
y = np.array(y)

# --
# Fit ASE

A = nx.to_numpy_array(G)

if args.mode == 'ase':
    embedder = AdjacencySpectralEmbed(algorithm='full')
elif args.mode == 'lse':
    embedder = LaplacianSpectralEmbed(algorithm='full', form='DAD')

X_hat = embedder.fit_transform(A)
X_hat = np.column_stack(X_hat)

# --
# Train classifiers

scores = np.zeros((n_class, args.n_iters))

for label_idx, label in enumerate(tqdm(ulabels)):
    for iter_idx in range(args.n_iters):
        
        X_train, X_test, y_train, y_test = train_test_split(
            X_hat,
            y == label,
            train_size=args.p_train,
            test_size=1 - args.p_train
        )
        
        model = BaggingClassifier(DecisionTreeClassifier())
        model = model.fit(X_train, y_train)
        y_hat = model.predict(X_test)
        
        scores[label_idx,iter_idx] = metrics.f1_score(y_test, y_hat, average='binary')

print(f'mode={args.mode}')
print('f1.mean', scores.mean(axis=-1))
print('f1.std', scores.std(axis=-1))