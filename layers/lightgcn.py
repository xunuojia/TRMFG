# coding=utf-8

import torch
from torch import nn
import torch.nn.functional as F
import torch_geometric

from torch_geometric.nn import LGConv



import warnings
warnings.filterwarnings("ignore",category=DeprecationWarning)



class Light(torch.nn.Module):
    def __init__(self,num_nodes,embedding_dim,num_layers,alpha=None,**kwargs):
        super(Light, self).__init__()
        
        self.num_nodes = num_nodes
        self.embedding_dim = embedding_dim
        self.num_layers = num_layers
        # self.dropx = torch.nn.Dropout(0.5)

        if alpha is None:
            alpha = 1. / (num_layers + 1)

        if isinstance(alpha, torch.Tensor):
            assert alpha.size(0) == num_layers + 1
        else:
            alpha = torch.tensor([alpha] * (num_layers + 1))
        self.register_buffer('alpha', alpha)

        self.embedding = torch.nn.Embedding(num_nodes, embedding_dim)
        self.convs = torch.nn.ModuleList([LGConv(**kwargs) for _ in range(num_layers)])

        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.embedding.weight)
        for conv in self.convs:
            conv.reset_parameters()


    def forward(self, edge_index):
        x = self.embedding.weight
        # x = self.dropx(x)

        out = x * self.alpha[0]

        for i in range(self.num_layers):
            x = self.convs[i](x, edge_index)
            # x = self.dropx(x)
            out = out + x * self.alpha[i + 1]

        return out


    


class LightModel(torch.nn.Module):
    def __init__(self,q_e,whole_num_tags,emb_size,layers,que_drop_rate):
        super(LightModel, self).__init__()
        self.emb_size =  emb_size
        self.whole_num_tags = whole_num_tags
        self.layers = layers
        self.q_emb = torch.nn.Parameter(torch.FloatTensor(q_e), requires_grad = False)
        # nn.init.xavier_uniform_(self.q_emb)
        # self.q_emb = torch.nn.Parameter(torch.FloatTensor(q_e), requires_grad = False)
        # self.t_emb = torch.nn.Parameter(torch.FloatTensor(t_e), requires_grad = False)
        self.light = Light(whole_num_tags,emb_size,layers)
       
        self.drop_q = torch.nn.Dropout(que_drop_rate)
        self.q_encoder = torch.nn.Sequential(
            torch.nn.Dropout(que_drop_rate),
            torch.nn.Linear(emb_size,emb_size)
            )

    def forward(self, tag_graph):
        x, edge_index = tag_graph.x, tag_graph.edge_index
        # h_edge = self.drop_edge(edge_index)
        # h = self.drop_layer(x)
        h = self.light(edge_index)

        ques = self.q_encoder(self.q_emb)
        # h = h1 + h2 + h3
        return ques, h
