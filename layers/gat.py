# coding=utf-8

# import torch
# import torch.nn.functional as F
# from transformers import AutoModel, AutoTokenizer, BertForMaskedLM
import torch
from torch import nn
import dgl
from torch_geometric.nn import GATConv
import dgl.nn.pytorch as dglnn
import torch.nn.functional as F

import warnings
warnings.filterwarnings("ignore",category=DeprecationWarning)



class GATModel(torch.nn.Module):
    def __init__(self,q_e,in_size,hid_size,heads,tag_drop_rate=0.0,que_drop_rate=0.0):
        super(GATModel, self).__init__()
        self.in_size = in_size
        self.hid_size = hid_size
        self.heads = heads
        self.q_emb = torch.nn.Parameter(torch.FloatTensor(q_e), requires_grad = False)
        # nn.init.xavier_uniform_(self.q_emb)
        
        # self.q_emb = torch.nn.Parameter(torch.FloatTensor(q_e), requires_grad = False)
        # self.t_emb = torch.nn.Parameter(torch.FloatTensor(t_e), requires_grad = False)
        self.gat1 = GATConv(in_size,hid_size,heads,tag_drop_rate)
        self.gat2 = GATConv(hid_size*heads,hid_size,heads,tag_drop_rate)
        
        self.drop_rate = tag_drop_rate
        self.drop_layer = torch.nn.Dropout(tag_drop_rate)
        self.drop_q = torch.nn.Dropout(que_drop_rate)
        self.q_encoder = torch.nn.Sequential(
            torch.nn.Dropout(que_drop_rate),
            torch.nn.Linear(in_size,hid_size*heads)
            )
        


    def forward(self,tag_graph):
        x, edge_index = tag_graph.x, tag_graph.edge_index
        # h_edge = self.drop_edge(edge_index)
        h = self.drop_layer(x)
        # h = self.conv1(h, edge_index)
        h = self.gat1(h, edge_index)
        h = F.relu(h)
        h = self.drop_layer(h)
        
        h = self.gat2(h, edge_index)
        # h = self.drop_layer(h)
        # h = self.gat3(h, edge_index)
        # h = self.conv2(h, edge_index)
        # h = self.drop_layer(h)

        ques = self.q_encoder(self.q_emb)
        # h = h + x
        return ques, h
