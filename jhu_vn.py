#!/usr/bin/env python

"""
    jhu_vn.py
"""

import json
import numpy as np
import networkx as nx
from tqdm import tqdm

from sklearn import metrics
from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split

from graspy.embed import AdjacencySpectralEmbed as ASE

# --
# Helpers

def to_coarse_labels(fine_labels):
    
    coarse_labels = -1 * np.ones(len(fine_labels), dtype=np.int)
    for i, label in enumerate(fine_labels):
        
        if "KC" in label:
            coarse_labels[i] = 0
        
        elif "PN" in label:
            coarse_labels[i] = 1
        
        elif "MBIN" == label:
            coarse_labels[i] = 2
            
        elif "MBON" == label:
            coarse_labels[i] = 3
            
        else:
            coarse_labels[i] = 4
            
    return coarse_labels

# --
# IO

inpath = 'data/maggot/mb_2019-09-23/G.graphml'

G = nx.read_graphml(inpath)

# --
# Cleaning

label_attribute   = "Class"
coarse_labels_all = ["KC", "PN", "MBIN", "MBON"]

fine_labels   = np.array(list(nx.get_node_attributes(G, label_attribute).values()))
coarse_labels = to_coarse_labels(fine_labels)
unique_coarse_labels = np.unique(coarse_labels)

# --
# Fit ASE

np.random.seed(1)

A = nx.to_numpy_array(G)

X_hat = ASE().fit_transform(A)
X_hat = np.concatenate(X_hat, axis=1)

n, d = X_hat.shape

# --
# Train classifiers

K = len(unique_coarse_labels[:-1])

prop_training = 0.1
num_iters     = 32

# precisions = np.zeros((len(unique_coarse_labels) - 1, num_iters))
# recalls    = np.zeros((len(unique_coarse_labels) - 1, num_iters))
f1s        = np.zeros((len(unique_coarse_labels) - 1, num_iters))

for i, label in enumerate(tqdm(unique_coarse_labels[:-1])):
    for j in range(num_iters):
        
        y = coarse_labels == label
        X_train, X_test, y_train, y_test = train_test_split(X_hat, y, train_size=prop_training)
        
        model = BaggingClassifier(DecisionTreeClassifier())
        model = model.fit(X_train, y_train)
        y_hat = model.predict(X_test)
        
        # precisions[i,j] = metrics.precision_score(y_test, y_hat)
        # recalls[i,j]    = metrics.recall_score(y_test, y_hat)
        f1s[i,j]        = metrics.f1_score(y_test, y_hat, average='binary')


print('f1.mean', f1s.mean(axis=-1))
print('f1.std', f1s.std(axis=-1))