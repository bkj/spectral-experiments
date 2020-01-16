#!/usr/bin/env python

"""
    embedders.py
"""

import sys
import numpy as np

import torch
from torch import nn
from torch.nn import functional as F

from ez_ppnp.models import EmbeddingPPNP
from ez_ppnp.trainer import train_unsupervised
from ez_ppnp.ppr import exact_ppr, exact_ppr_joblib, PrecomputedPPR

from graspy.embed import AdjacencySpectralEmbed, LaplacianSpectralEmbed

from helpers import to_numpy

def embed_ppnp(adj, ppr_alpha, hidden_dim, lr, epochs, batch_size):
    n_nodes = adj.shape[0]
    
    ppr_fn    = exact_ppr_joblib if n_nodes > 5000 else exact_ppr
    ppr_array = ppr_fn(adj, alpha=ppr_alpha)
    np.fill_diagonal(ppr_array, 0)
    
    # --
    # Train embedding
    
    model = EmbeddingPPNP(
        ppr        = PrecomputedPPR(ppr=ppr_array),
        n_nodes    = n_nodes,
        hidden_dim = hidden_dim,
    )
    
    model = model.cuda()
    model = model.train()
    
    loss_hist = train_unsupervised(model, lr=lr, epochs=epochs, batch_size=batch_size)
    
    # --
    # Compute X_hat
    
    idx_chunks = np.array_split(np.arange(n_nodes), n_nodes // batch_size)
    with torch.no_grad():
        X_hat = np.row_stack([to_numpy(model(idx_chunk)[1]) for idx_chunk in idx_chunks])
    
    return X_hat


def embed_ase(adj, n_components=None):
    if n_components is not None:
        print(n_components)
    
    X_ase = AdjacencySpectralEmbed(n_components=n_components).fit_transform(adj.toarray())
    if isinstance(X_ase, tuple):
        X_ase = np.column_stack(X_ase)
    
    return X_ase


def embed_lse(adj, n_components=None):
    if n_components is not None:
        print(n_components)
    
    X_lse = LaplacianSpectralEmbed(n_components=n_components).fit_transform(adj.toarray())
    if isinstance(X_lse, tuple):
        X_lse = np.column_stack(X_lse)
    
    return X_lse

