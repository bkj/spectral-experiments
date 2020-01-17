#!/usr/bin/env python

"""
    datasets/desikan.py
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd
import networkx as nx
from scipy import sparse as sp

from graspy.utils import pass_to_ranks

sys.path.append('.')
from helpers import set_seeds, save_csr

# --
# CLI

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--label-inpath', type=str,   default='./data/DS72784/DS72784_desikan.csv')
    parser.add_argument('--inpath',       type=str,   default='./data/DS72784/subj1-scan1')
    parser.add_argument('--seed',         type=int,   default=123)
    return parser.parse_args()


args = parse_args()
set_seeds(args.seed)

# --
# IO

print(f'prep_desikan.py: loading {args.inpath}', file=sys.stderr)

G = nx.read_graphml(args.inpath + '.graphml')

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
# Preprocess

print(f'prep_desikan.py: computing A', file=sys.stderr)
A      = nx.to_numpy_array(G)

print(f'prep_desikan.py: computing A_ptr', file=sys.stderr)
A_ptr  = pass_to_ranks(A, method='simple-nonzero')


# --
# Save

print(f'prep_desikan.py: saving to {args.inpath}', file=sys.stderr)

save_csr(args.inpath + '.adj.npy', A_ptr)
np.save(args.inpath + '.y.npy', y)

