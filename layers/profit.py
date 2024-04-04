# coding=utf-8


import torch
from torch import nn


import dgl
import dgl.nn as dglnn
import torch.autograd as autograd
import torch_geometric
from torch_geometric.nn import GCNConv
from torch_geometric.data import Data
import torch.nn.functional as F
import dgl.function as fn
from dgl.nn.functional import edge_softmax
# from dgl.nn import HeteroEmbedding
import warnings
warnings.filterwarnings("ignore",category=DeprecationWarning)




class PROFIT(torch.nn.Module):
    def __init__(self,q_e,in_size,out_size,cached=False,tag_drop_rate=0.0,que_drop_rate=0.0):
        super(PROFIT, self).__init__()
        self.in_size = in_size
        self.out_size = out_size
        
        self.q_emb = torch.nn.Parameter(torch.FloatTensor(q_e), requires_grad = False)
        # nn.init.xavier_uniform_(self.q_emb)
        # self.t_emb = torch.nn.Parameter(torch.FloatTensor(t_e), requires_grad = False)
        self.conv1 = GCNConv(in_size,out_size,cached=cached)
        self.conv2 = GCNConv(out_size,out_size,cached=cached)#cache true
        self.drop_rate = tag_drop_rate
        self.drop_layer = torch.nn.Dropout(tag_drop_rate)
        self.drop_q = torch.nn.Dropout(que_drop_rate)
        # self.q_encoder = torch.nn.Sequential(
        #     torch.nn.Dropout(que_drop_rate),
        #     torch.nn.Linear(in_size,out_size)
        #     )
        self.mlp = nn.Linear(self.in_size,self.out_size)
        self.layers = []
        for layer in range(1):
            convs = []
            for filt in range(3,6):
                in_channels =  self.in_size
                out_channels = self.out_size
                kernel_size = filt
                new_conv = nn.Conv1d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size)
                self.add_module( 'layer_'+str(layer)+'_conv_'+str(filt), new_conv)
                convs.append(new_conv)

            self.layers.append(convs)

        # self.max_pool = max_pool_over_time
        self.device = None

        
        
    def forward(self, tag_graph):
        x, edge_index = tag_graph.x, tag_graph.edge_index
        # h_edge = self.drop_edge(edge_index)
        h = self.drop_layer(x)
        h1 = self.conv1(h, edge_index)
        h1 = F.relu(h1)
        h1 = self.drop_layer(h1)
        h2 = self.conv2(h1, edge_index)
        # h2 = F.relu(h2)
        # h2 = self.drop_layer(h2)
        # h3 = self.conv3(h2, edge_index)
        self.device = x.device

        layer_activ = self.q_emb
        for layer in self.layers:
            next_activ = []
            for conv in layer:
                padded_activ = self.mlp(layer_activ)
                next_activ = padded_activ

            # concat across channels
            layer_activ = F.relu(next_activ)

        ques = layer_activ
        # print(ques.shape)



        # ques = self.q_encoder(self.q_emb)
        h = h2# + h3
        return ques, h