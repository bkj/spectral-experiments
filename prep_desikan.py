#!/usr/bin/env python

"""
    jhu_vn_desikan.py
"""

import sys
import argparse
import numpy as np
import pandas as pd
import networkx as nx
from scipy import sparse as sp

from graspy.utils import pass_to_ranks

from helpers import set_seeds, save_csr, train_stop_valid_split

# --
# CLI

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--graph-inpath',   type=str,   default='./data/DS72784/subj1-scan1.graphml')
    parser.add_argument('--graph-outpath',  type=str,   default='./data/DS72784/subj1-scan1.A_ptr.npy')
    
    parser.add_argument('--label-inpath',   type=str,   default='./data/DS72784/DS72784_desikan.csv')
    parser.add_argument('--label-outpath',  type=str,   default='./data/DS72784/subj1-scan1.y.npy')
    
    parser.add_argument('--seed',          type=int,   default=123)
    return parser.parse_args()


args = parse_args()
set_seeds(args.seed)

# --
# IO

print(f'prep_desikan.py: loading {args.graph_inpath}', file=sys.stderr)

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
# Preprocess

print(f'prep_desikan.py: computing A', file=sys.stderr)
A      = nx.to_numpy_array(G)

print(f'prep_desikan.py: computing A_ptr', file=sys.stderr)
A_ptr  = pass_to_ranks(A, method='simple-nonzero')


# --
# Save

print(f'prep_desikan.py: saving to {args.graph_outpath}', file=sys.stderr)
save_csr(args.graph_outpath, A_ptr)

print(f'prep_desikan.py: saving to {args.label_outpath}', file=sys.stderr)
np.save(args.label_outpath, y)

