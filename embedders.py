#!/usr/bin/env python

"""
    embedders.py
"""

import numpy as np
from sklearn.utils.extmath import randomized_svd

import torch
from torch import nn
from torch.nn import functional as F

from graspy.embed import AdjacencySpectralEmbed, LaplacianSpectralEmbed

from ez_ppnp.models import EmbeddingPPNP, SupervisedEmbeddingPPNP
from ez_ppnp.trainer import train_unsupervised, train_supervised
from ez_ppnp.ppr import exact_ppr, exact_ppr_joblib, PrecomputedPPR

from helpers import to_numpy

def smart_ppr(adj, alpha):
    n_nodes = adj.shape[0]
    ppr_fn  = exact_ppr_joblib if n_nodes > 5000 else exact_ppr
    return ppr_fn(adj, alpha=alpha)


def embed_ppnp(*, ppr_array, hidden_dim, lr, epochs, batch_size):
    """ minimize l2 distance between node embedding and ppr-weighted average of other node embeddings"""
    
    n_nodes = ppr_array.shape[0]
    
    np.fill_diagonal(ppr_array, 0)
    
    model = EmbeddingPPNP(
        ppr        = PrecomputedPPR(ppr=ppr_array),
        n_nodes    = n_nodes,
        hidden_dim = hidden_dim,
    )
    
    model = model.cuda()
    model = model.train()
    
    loss_hist = train_unsupervised(model, lr=lr, epochs=epochs, batch_size=batch_size)
    
    idx_chunks = np.array_split(np.arange(n_nodes), n_nodes // batch_size)
    with torch.no_grad():
        X_hat = np.row_stack([to_numpy(model(idx_chunk)[1]) for idx_chunk in idx_chunks])
    
    return X_hat


def embed_ppnp_supervised(*, ppr_array, y, idx_train, hidden_dim, lr, epochs, batch_size):
    """ minimize cross entropy between linear transform of node embedding and labels.  node embedding is ppr-weights average of other node embedding. """
    
    # !! Could benefit a lot from early stopping
    # !! Could benefit a lot from features
    
    n_nodes = ppr_array.shape[0]
    
    # --
    # Train embedding
    
    # np.fill_diagonal(ppr_array, 0)
    
    model = SupervisedEmbeddingPPNP(
        ppr        = PrecomputedPPR(ppr=ppr_array),
        n_nodes    = n_nodes,
        hidden_dim = hidden_dim,
        n_classes  = len(set(y))
    )
    
    model = model.cuda()
    model = model.train()
    
    loss_hist = train_supervised(model, y, idx_train, lr=lr, epochs=epochs, batch_size=batch_size)
    
    idx_chunks = np.array_split(np.arange(n_nodes), n_nodes // batch_size)
    with torch.no_grad():
        X_hat = np.row_stack([to_numpy(model(idx_chunk)) for idx_chunk in idx_chunks])
    
    return X_hat


def embed_ppr_svd(*, ppr_array, n_components, topk=None):
    """ svd on ppr_array. if topk != None, sparsify ppr_array by dropping all but topk values"""
    
    if topk is not None:
        threshes = np.sort(ppr_array, axis=-1)[:,-topk] # !! Slow -- should replace w/ argpartition
        ppr_array[ppr_array < threshes.reshape(-1, 1)] = 0
    
    u, _, _  = randomized_svd(ppr_array, n_components=n_components)
    return u


def embed_ase(*, adj, n_components=None):
    """ JHU AdjacencySpectralEmbed, w/ default settings """
    X_ase = AdjacencySpectralEmbed(n_components=n_components).fit_transform(adj.toarray())
    if isinstance(X_ase, tuple):
        X_ase = np.column_stack(X_ase)
    
    return X_ase


def embed_lse(*, adj, n_components=None):
    """ JHU LaplacianSpectralEmbed, w/ default settings """
    X_lse = LaplacianSpectralEmbed(n_components=n_components).fit_transform(adj.toarray())
    if isinstance(X_lse, tuple):
        X_lse = np.column_stack(X_lse)
    
    return X_lse

