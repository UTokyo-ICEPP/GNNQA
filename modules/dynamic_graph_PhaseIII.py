import numpy as np
import pandas as pd
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F_n

import os, sys

import dgl
from dgl import backend as F
import dgl.function as fn
from dgl.nn.pytorch import KNNGraph

from modules.mlp import build_mlp
import networkx as nx
import matplotlib.pyplot as plt
import math

DEBUG = False

#cuda_device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu' )

# ---------------- The EdgeConv function ---------------------- #
class EdgeConv(nn.Module):

    def __init__(self,
                 in_feat_x,
                 out_feat_x,
                 in_feat_en,
                 out_feat_en,
                 in_feat_D_en,
                 out_feat_D_en,
                 batch_norm=False,
                 k_val=10):
        super(EdgeConv, self).__init__()

        self.batch_norm = batch_norm

        self.k = k_val

        self.theta = nn.Linear(in_feat_x, out_feat_x)

        self.phi = nn.Linear(in_feat_x, out_feat_x)

        self.theta_en = build_mlp(inputsize  = in_feat_en,\
                                  outputsize = out_feat_en,\
                                  features = [32, 64, 64, 64, 32],\
                                  add_batch_norm = batch_norm
                                  )

        self.phi_en   = build_mlp(inputsize  = in_feat_en,\
                                  outputsize = out_feat_en,\
                                  features = [32, 64, 64, 64, 32],\
                                  add_batch_norm = batch_norm
                                  )


        self.W      = build_mlp(inputsize  = (2*in_feat_x+2*in_feat_en),\
                                  outputsize = out_feat_D_en,\
                                  features = [32, 64, 64, 64, 32],\
                                  add_batch_norm = batch_norm
                                  )


    def message(self, edges):
        """The message computation function.
        """

        theta_x = self.theta(edges.dst['x'] - edges.src['x'])
        phi_x = self.phi(edges.dst['x'])

        theta_en = self.theta_en(edges.dst['en'] - edges.src['en'])
        phi_en = self.phi_en(edges.dst['en'])

        w_input_Alt = torch.cat([edges.src['x'], edges.src['en'],  edges.dst['x'], edges.dst['en']], 1)

        return { 'm_x': theta_x+phi_x,
                 'm_en': theta_en+phi_en,
                 'm_edge': w_input_Alt }

    def forward(self, g):
        """Forward computation"""

        #First copy the features from g
        idx_data = g.ndata['idx']
        x_data = g.ndata['x']
        en_data = g.ndata['en']

        n_part = len(x_data)

        #self.k = int(n_part*0.25)
        #self.k = int(math.sqrt(n_part))
        self.k = 5

        del g; torch.cuda.empty_cache();# del g -> only keep the features

        transform = dgl.RemoveSelfLoop()

        g_new_o = dgl.knn_graph(x_data, self.k)
        g_new = dgl.remove_self_loop(g_new_o)

        #setup the node features for the knn graph
        g_new.ndata['idx'] = idx_data
        g_new.ndata['x'] = x_data
        g_new.ndata['en'] = en_data
        #g_new.edata['m_edge'] = edge_data
        #g_new.edata['score'] = edge_data

        #update edge and node features
        g_new.apply_edges(self.message)
        g_new.update_all(fn.copy_e('m_x', 'm_x'), fn.mean('m_x', 'x')) # "x" is updated
        g_new.update_all(fn.copy_e('m_en', 'm_en'), fn.mean('m_en', 'en')) # "en" is updated
        if 1:
            g_new.edata['score'] = self.W(g_new.edata['m_edge']) # "score" is calculated
        else:
            isScoreTerm = 0
            for attr in g.edge_attr_schemes():
                if attr == 'score':
                    isScoreTerm = 1
                    break

            if DEBUG:
                print("isScoreTerm:",isScoreTerm)

            if isScoreTerm == 0:
                g_new.edata['score'] = self.W(g_new.edata['m_edge']) # "score" is calculated
                if DEBUG:
                    print ("isScoreTerm == 0, g.edata['score']: ", g_new.edata['score'])
            else:
                g_new.edata['score'] = g_new.edata['score']+self.W(g_new.edata['m_edge']) # "score" is calculated
                if DEBUG:
                    print ("isScoreTerm != 0, g.edata['score']: ", g_new.edata['score'])

        NumEdge = g_new.number_of_edges()
        return g_new, g_new.edata['score']

class Dynamic_Graph_Model(nn.Module):
    def __init__(self, nlayers=3, nclass=1, nodes=16, device=torch.device("cpu")):
        super(Dynamic_Graph_Model, self).__init__()

        self.device = device

        self.n_layers = nlayers

        self.embedding_position = nn.Linear(3, 8) # node 3 -> 8
        self.embedding_node = nn.Linear(2, nodes) # node 4 -> nodes(16)
        #self.embedding_edge = nn.Linear(4, nodes) # edge 3 -> nodes(16)

        self.layer_list = nn.ModuleList()

        for i in range(self.n_layers):
            self.layer_list.append(
                EdgeConv(in_feat_x = 8, out_feat_x = 8,\
                         in_feat_en = nodes, out_feat_en = nodes,\
                         in_feat_D_en = nodes, out_feat_D_en = nodes,\
                         batch_norm = True)
            )

        self.decording = nn.Linear(nodes,nclass)

        self.act = nn.Softmax(dim=0)

    # -- the forward function -- #
    def forward(self, g):
        if 1:
            with g.local_scope():
                score_array = torch.empty((0)).to(self.device)
                edge_array = []
                node_array = []
                #graph_list = dgl.unbatch(g)
                graph_list = [g]
                for ig in graph_list: #Here we are using graph_list = 1 graph...
                    ig.ndata['x'] = self.embedding_position(ig.ndata['x'])
                    ig.ndata['en'] = self.embedding_node(ig.ndata['en'])

                    for iEC in range(self.n_layers-1):
                        ig, _ = self.layer_list[iEC](ig)

                    # the final one
                    _, score = self.layer_list[self.n_layers-1](ig)

                    #target graph, in sone case the selfremoval in knn-graph will causing the number of edges between ig and _ didn't match
                    tg = _

                    #print ("tg",tg)

                    h = self.decording(score).to(self.device)
                    if DEBUG:
                        print ("h", h)
                        print ("tg",tg)
                        print ("score",score.shape)
                        print("h",h.shape)
                        print("x",tg.ndata['x'].shape)
                        print("en",tg.ndata['en'].shape)
                        print("idx",tg.ndata['idx'].shape)
                    score_array = h
                    edge_array.append(tg.edges())

                    node_array.append(tg.ndata['idx'])
                    if DEBUG:
                        print ("score_array: ",score_array.shape)
                        print ("edge_array: ",edge_array)
                        print ("edge_array: ",edge_array[0][0].shape)

                return score_array,edge_array,node_array
