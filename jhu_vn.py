#!/usr/bin/env python

"""
    jhu_vn.py
"""

import numpy as np
import networkx as nx

from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier
from graspy.embed import AdjacencySpectralEmbed as ASE

from sklearn import metrics
​

file='data/maggot//mb_2019-09-23/G.graphml'

G = nx.read_graphml(file)

​

label_attribute="Class"

coarse_labels_all=["KC", "PN", "MBIN", "MBON"]

​

fine_labels = np.array(list(nx.get_node_attributes(G, label_attribute).values()))

unique_fine_labels = np.unique(y)

def to_coarse_labels(fine_labels):

    coarse_labels=-1*np.ones(len(fine_labels))

    for i, label in enumerate(fine_labels):

        if "KC" in label:

            coarse_labels[i]=0

        elif "PN" in label:

            coarse_labels[i]=1

        elif "MBIN" == label:

            coarse_labels[i]=2

        elif "MBON" == label:

            coarse_labels[i]=3

        else:

            coarse_labels[i]=4

    return coarse_labels

coarse_labels=to_coarse_labels(fine_labels)

​

np.random.seed(1)

A = nx.to_numpy_array(G)

​

X_hat = np.concatenate(ASE().fit_transform(A), axis=1)

n,d = X_hat.shape

​

unique_coarse_labels=np.unique(coarse_labels)

K=len(unique_coarse_labels[:-1])

​

precisions=np.zeros(K)

recalls=np.zeros(K)

f1=np.zeros(K)

​

for i, label in enumerate(unique_coarse_labels[:-1]):

    prop_training = 0.1

    train_idx = np.random.choice(np.arange(n), np.math.ceil(n*prop_training))

    

    temp_labels = (coarse_labels == label).astype(int)

    model = BaggingClassifier(DecisionTreeClassifier())

​

    test_idx = np.array([i for i in range(n) if i not in train_idx])

​

    X_train, y_train=X_hat[train_idx], temp_labels[train_idx]

    X_test, y_test=X_hat[test_idx], temp_labels[test_idx]

        

    model.fit(X_train, y_train)

    y_hat=model.predict(X_test)

    

    precisions[i]=precision_score(y_test, y_hat)

    recalls[i]=recall_score(y_test, y_hat)

    f1[i]=f1_score(y_test, y_hat)

​

for i in range(K):

    print(coarse_labels_all[i], f1[i])

