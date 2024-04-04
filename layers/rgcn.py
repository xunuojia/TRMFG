# coding=utf-8


import torch
from torch import nn
import dgl

import dgl.nn.pytorch as dglnn
import torch.nn.functional as F
import dgl.function as fn


import warnings
warnings.filterwarnings("ignore",category=DeprecationWarning)




class HeteroRGCNLayer(nn.Module):
    def __init__(self, in_size, out_size, etypes):
        super(HeteroRGCNLayer, self).__init__()
        # W_r for each relation
        self.weight = nn.ModuleDict(
            {name: nn.Linear(in_size, out_size) for name in etypes}
        )

    def forward(self, G, feat_dict):
        # The input is a dictionary of node features for each type
        funcs = {}
        for srctype, etype, dsttype in G.canonical_etypes:
            # Compute W_r * h
            Wh = self.weight[etype](feat_dict[srctype])
            # Save it in graph for message passing
            G.nodes[srctype].data["Wh_%s" % etype] = Wh
            
            funcs[etype] = (fn.copy_u("Wh_%s" % etype, "m"), fn.mean("m", "h"))
       
        G.multi_update_all(funcs, "sum")
        # return the updated node feature dictionary
        return {ntype: G.nodes[ntype].data["h"] for ntype in G.ntypes}


class HeteroRGCN(nn.Module):
    def __init__(self, in_size, hidden_size, out_size):
        super(HeteroRGCN, self).__init__()
        # create layers
        self.layer1 = HeteroRGCNLayer(in_size, hidden_size, {'give','inter_1','inter_2'})
        self.layer2 = HeteroRGCNLayer(hidden_size, out_size, {'give','inter_1','inter_2'})

    def forward(self, G, out_key1,out_key2,h):
        input_dict = h
        h_dict = self.layer1(G, input_dict)
        h_dict = {k: F.leaky_relu(h) for k, h in h_dict.items()}
        h_dict = self.layer2(G, h_dict)
        # get appropriate logits
        return h_dict[out_key1], h_dict[out_key2]

class RGCNModel(torch.nn.Module):
    def __init__(self,q_e,t_e,in_size,hid_size,out_size):
        super(RGCNModel, self).__init__()
        
        self.hgt = HeteroRGCN(in_size,hid_size,out_size)
        self.q_emb = torch.nn.Parameter(torch.FloatTensor(q_e), requires_grad = False)
        # nn.init.xavier_uniform_(self.q_emb)
        self.t_emb = torch.nn.Parameter(torch.FloatTensor(t_e), requires_grad = False)
            
        


    def forward(self, G):
        h = {}
        h['tag'] = self.t_emb
        h['question']= self.q_emb
        

        tag,ques = self.hgt(G,'tag','question',h)
        # ques = self.q_encoder(question)
        # out = out + x
        return ques, tag
       





