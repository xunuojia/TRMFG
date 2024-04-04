# coding=utf-8

import torch
from torch import nn
import dgl
from torch_geometric.nn import GCNConv,APPNP,GATConv
import dgl.nn.pytorch as dglnn
import torch.nn.functional as F
import dgl.function as fn



import warnings
warnings.filterwarnings("ignore",category=DeprecationWarning)




class APPNPModel(torch.nn.Module):
    def __init__(self,q_e,in_size,out_size,cached=False,k=2,alpha=0.1,tag_drop_rate=0.0,que_drop_rate=0.0):
        super(APPNPModel, self).__init__()
        self.in_size = in_size
        self.out_size = out_size
        self.k = k
        self.alpha = alpha
        self.q_emb = torch.nn.Parameter(torch.FloatTensor(q_e), requires_grad = False)
        # nn.init.xavier_uniform_(self.q_emb)
        
        # self.q_emb = torch.nn.Parameter(torch.FloatTensor(q_e), requires_grad = False)
        # self.t_emb = torch.nn.Parameter(torch.FloatTensor(t_e), requires_grad = False)
        self.appnp1 = APPNP(K=2,alpha=0.1,dropout=0.1,cached=cached)#cached true
        self.drop_rate = tag_drop_rate
        self.drop_layer = torch.nn.Dropout(tag_drop_rate)
        self.drop_q = torch.nn.Dropout(que_drop_rate)
        self.q_encoder = torch.nn.Sequential(
            torch.nn.Dropout(que_drop_rate),
            torch.nn.Linear(in_size,out_size)
            )
        self.t_encoder = torch.nn.Linear(in_size,out_size)

    def forward(self,tag_graph):
        x, edge_index = tag_graph.x, tag_graph.edge_index
        # h_edge = self.drop_edge(edge_index)
        x = self.t_encoder(x)
        h = self.drop_layer(x)
        h = self.appnp1(h, edge_index)
        # h = self.drop_layer(h)
        # h = self.conv2(h, edge_index)
        # h = self.drop_layer(h)

        ques = self.q_encoder(self.q_emb)
        
        return ques, h
