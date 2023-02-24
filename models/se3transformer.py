import sys

import numpy as np
import torch
from torch_scatter import scatter
from torch_geometric.nn import radius_graph

import dgl
from dgl.nn.pytorch import GraphConv, NNConv
from torch import nn
from torch.nn import functional as F
from typing import Dict, Tuple, List

from models.se3transformer_layers.modules import GConvSE3, GNormSE3, get_basis_and_r, GSE3Res, GMaxPooling, GAvgPooling
from models.se3transformer_layers.fibers import Fiber


class TFN(nn.Module):
    """SE(3) equivariant GCN"""
    def __init__(self, num_layers: int, atom_feature_size: int, 
                 num_channels: int, num_nlayers: int=1, num_degrees: int=4, 
                 edge_dim: int=4, **kwargs):
        super().__init__()
        # Build the network
        self.num_layers = num_layers
        self.num_nlayers = num_nlayers
        self.num_channels = num_channels
        self.num_degrees = num_degrees
        self.num_channels_out = num_channels*num_degrees
        self.edge_dim = edge_dim

        self.fibers = {'in': Fiber(1, atom_feature_size),
                       'mid': Fiber(num_degrees, self.num_channels),
                       'out': Fiber(1, self.num_channels_out)}

        blocks = self._build_gcn(self.fibers, 1)
        self.block0, self.block1, self.block2 = blocks

    def _build_gcn(self, fibers, out_dim):

        block0 = []
        fin = fibers['in']
        for i in range(self.num_layers-1):
            block0.append(GConvSE3(fin, fibers['mid'], self_interaction=True, edge_dim=self.edge_dim))
            block0.append(GNormSE3(fibers['mid'], num_layers=self.num_nlayers))
            fin = fibers['mid']
        block0.append(GConvSE3(fibers['mid'], fibers['out'], self_interaction=True, edge_dim=self.edge_dim))

        block1 = [GMaxPooling()]

        block2 = []
        block2.append(nn.Linear(self.num_channels_out, self.num_channels_out))
        block2.append(nn.ReLU(inplace=True))
        block2.append(nn.Linear(self.num_channels_out, out_dim))

        return nn.ModuleList(block0), nn.ModuleList(block1), nn.ModuleList(block2)

    def forward(self, G):
        # Compute equivariant weight basis from relative positions
        basis, r = get_basis_and_r(G, self.num_degrees-1)

        # encoder (equivariant layers)
        h = {'0': G.ndata['f']}
        for layer in self.block0:
            h = layer(h, G=G, r=r, basis=basis)

        for layer in self.block1:
            h = layer(h, G)

        for layer in self.block2:
            h = layer(h)

        return h


class SE3Transformer(nn.Module):
    """SE(3) equivariant GCN with attention"""
    def __init__(self, num_layers: int, atom_feature_size: int, 
                 num_channels: int, num_degrees: int=4, edge_dim: int=4, 
                 div: float=4, pooling: str='avg', n_heads: int=1, 
                 cutoff: float=5.0, max_num_neighbors: int=32, auto_grad: bool=False,
                 **kwargs):
        super().__init__()
        
        self.cutoff = cutoff
        self.max_num_neighbors = max_num_neighbors

        # Build the network
        self.num_layers = num_layers
        self.num_channels = num_channels
        self.num_degrees = num_degrees
        self.edge_dim = edge_dim
        self.div = div
        self.pooling = pooling
        self.n_heads = n_heads

        self.fibers = {'in': Fiber(1, atom_feature_size),
                       'mid': Fiber(num_degrees, self.num_channels),
                       'out': Fiber(1, num_degrees*self.num_channels)}

        blocks = self._build_gcn(self.fibers, 1)
        self.Gblock, self.FCblock = blocks

        self.auto_grad = auto_grad

    def _build_gcn(self, fibers, out_dim):
        # Equivariant layers
        Gblock = []
        fin = fibers['in']
        for i in range(self.num_layers):
            Gblock.append(GSE3Res(fin, fibers['mid'], edge_dim=self.edge_dim, 
                                  div=self.div, n_heads=self.n_heads))
            Gblock.append(GNormSE3(fibers['mid']))
            fin = fibers['mid']
        Gblock.append(GConvSE3(fibers['mid'], fibers['out'], self_interaction=True, edge_dim=self.edge_dim))

        # Pooling
        if self.pooling == 'avg':
            Gblock.append(GAvgPooling())
        elif self.pooling == 'max':
            Gblock.append(GMaxPooling())

        # FC layers
        FCblock = []
        FCblock.append(nn.Linear(self.fibers['out'].n_features, self.fibers['out'].n_features))
        FCblock.append(nn.ReLU(inplace=True))
        FCblock.append(nn.Linear(self.fibers['out'].n_features, out_dim))

        return nn.ModuleList(Gblock), nn.ModuleList(FCblock)

    def _build_graph(self, data):
        edge_index = radius_graph(
            data.pos,
            r=self.cutoff,
            loop=False,
            batch=data.batch,
            max_num_neighbors=self.max_num_neighbors
        )

        src = torch.cat([edge_index[0,:], edge_index[1,:]])
        dst = torch.cat([edge_index[1,:], edge_index[0,:]])

        G = dgl.graph((src, dst), num_nodes=data.pos.shape[0])
        G.ndata['x'] = data.pos # [num_atoms,3]
        G.ndata['f'] = data.f # [num_atoms,28,1]
        G.edata['w'] = torch.zeros((G.num_edges(), 4), dtype=torch.float) # [num_edges,4]

        G.set_batch_num_nodes(data.num_atoms)

        return G

    def forward(self, data, device):
        G = self._build_graph(data)
        G = G.to(device)
        if self.auto_grad:
            G.ndata['x'].requires_grad_(True)
        src = G.edges()[0]
        dst = G.edges()[1]
        G.edata['d'] = G.ndata['x'][dst] - G.ndata['x'][src] # [num_edges,3]

        # Compute equivariant weight basis from relative positions
        basis, r = get_basis_and_r(G, self.num_degrees-1)

        # encoder (equivariant layers)
        h = {'0': G.ndata['f']}
        for layer in self.Gblock:
            h = layer(h, G=G, r=r, basis=basis)

        for layer in self.FCblock:
            h = layer(h)
        
        if self.auto_grad:
            forces = -1 * (
                torch.autograd.grad(
                    h,
                    G.ndata['x'],
                    grad_outputs=torch.ones_like(h),
                    create_graph=True,
                )[0]
            )
        else:
            forces = None

        return h, forces