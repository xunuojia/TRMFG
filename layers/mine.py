# coding=utf-8

import math 
import torch
from torch import nn
import dgl

import dgl.nn.pytorch as dglnn
import torch.nn.functional as F
import dgl.function as fn
from dgl.nn.functional import edge_softmax
# from dgl.nn import HeteroEmbedding

import warnings
warnings.filterwarnings("ignore",category=DeprecationWarning)


class QTLayer(nn.Module):
    def __init__(self,
                 in_dim,
                 out_dim,
                 node_dict,
                 edge_dict,
                 n_heads,
                 dropout = 0.5,
                 use_norm = False):
        super(QTLayer, self).__init__()

        self.in_dim        = in_dim
        self.out_dim       = out_dim
        self.node_dict     = node_dict
        self.edge_dict     = edge_dict
        self.num_types     = len(node_dict)
        self.num_relations = len(edge_dict)
        self.total_rel     = self.num_types * self.num_relations * self.num_types
        self.n_heads       = n_heads
        self.d_k           = out_dim // n_heads
        self.sqrt_dk       = math.sqrt(self.d_k)
        self.att           = None

        self.k_linears   = nn.ModuleList()
        self.q_linears   = nn.ModuleList()
        self.v_linears   = nn.ModuleList()
        self.a_linears   = nn.ModuleList()
        self.norms       = nn.ModuleList()
        self.use_norm    = use_norm

        for t in range(self.num_types):
            self.k_linears.append(nn.Linear(in_dim,   out_dim))
            self.q_linears.append(nn.Linear(in_dim,   out_dim))
            self.v_linears.append(nn.Linear(in_dim,   out_dim))
            self.a_linears.append(nn.Linear(out_dim,  out_dim))
            if use_norm:
                self.norms.append(nn.LayerNorm(out_dim))

        self.relation_pri   = nn.Parameter(torch.ones(self.num_relations, self.n_heads))
        self.relation_att   = nn.Parameter(torch.Tensor(self.num_relations, n_heads, self.d_k, self.d_k))
        self.relation_msg   = nn.Parameter(torch.Tensor(self.num_relations, n_heads, self.d_k, self.d_k))
        self.skip           = nn.Parameter(torch.ones(self.num_types))
        self.drop           = nn.Dropout(dropout)

        nn.init.xavier_uniform_(self.relation_att)
        nn.init.xavier_uniform_(self.relation_msg)

    def forward(self, G, h):
        with G.local_scope():
            node_dict, edge_dict = self.node_dict, self.edge_dict
            for srctype, etype, dsttype in G.canonical_etypes:
                # print(srctype,dsttype)
                if srctype == dsttype:

                    sub_graph = G[srctype, etype, dsttype]

                    k_linear = self.k_linears[node_dict[srctype]]
                    v_linear = self.v_linears[node_dict[srctype]]
                    q_linear = self.q_linears[node_dict[dsttype]]

                    k = k_linear(h[srctype]).view(-1, self.n_heads, self.d_k)
                    v = v_linear(h[srctype]).view(-1, self.n_heads, self.d_k)
                    q = q_linear(h[dsttype]).view(-1, self.n_heads, self.d_k)

                    e_id = self.edge_dict[etype]

                    relation_att = self.relation_att[e_id]
                    relation_pri = self.relation_pri[e_id]
                    relation_msg = self.relation_msg[e_id]

                    k = torch.einsum("bij,ijk->bik", k, relation_att)
                    v = torch.einsum("bij,ijk->bik", v, relation_msg)

                    sub_graph.srcdata['k'] = k
                    sub_graph.dstdata['q'] = q
                    sub_graph.srcdata['v_%d' % e_id] = v

                    sub_graph.apply_edges(fn.v_dot_u('q', 'k', 't'))
                    attn_score = sub_graph.edata.pop('t').sum(-1) * relation_pri / self.sqrt_dk
                    attn_score = edge_softmax(sub_graph, attn_score, norm_by='dst')

                    sub_graph.edata['t'] = attn_score.unsqueeze(-1)

                else:
                    sub_graph = G[srctype, etype, dsttype]

                    k_linear = self.k_linears[node_dict[srctype]]
                    v_linear = self.v_linears[node_dict[srctype]]
                    q_linear = self.q_linears[node_dict[dsttype]]

                    k = k_linear(h[srctype]).view(-1, self.n_heads, self.d_k)
                    v = v_linear(h[srctype]).view(-1, self.n_heads, self.d_k)
                    q = q_linear(h[dsttype]).view(-1, self.n_heads, self.d_k)

                    e_id = self.edge_dict[etype]

                    relation_att = self.relation_att[e_id]
                    relation_pri = self.relation_pri[e_id]
                    relation_msg = self.relation_msg[e_id]

                    k = torch.einsum("bij,ijk->bik", k, relation_att)
                    v = torch.einsum("bij,ijk->bik", v, relation_msg)

                    sub_graph.srcdata['k'] = k
                    sub_graph.dstdata['q'] = q
                    sub_graph.srcdata['v_%d' % e_id] = v

                    sub_graph.apply_edges(fn.v_dot_u('q', 'k', 't'))
                    attn_score = sub_graph.edata.pop('t').sum(-1) * relation_pri / self.sqrt_dk
                    attn_score = edge_softmax(sub_graph, attn_score, norm_by='dst')

                    sub_graph.edata['t'] = attn_score.unsqueeze(-1)

            G.multi_update_all({etype : (fn.u_mul_e('v_%d' % e_id, 't', 'm'), fn.sum('m', 't')) \
                                for etype, e_id in edge_dict.items()}, cross_reducer = 'mean')

            new_h = {}

            # print(G.nodes['tag'].data.keys())
            # print(G.nodes['question'].data.keys())
            for ntype in G.ntypes:
                
                n_id = node_dict[ntype]
                # alpha = torch.sigmoid(self.skip[n_id])
                # print(alpha)
                # print(G.nodes[ntype].data.keys())
                # print(G.edges[etype].data.keys())
                t = G.nodes[ntype].data['t'].view(-1, self.out_dim)
                # trans_out = self.drop(self.a_linears[n_id](t))
                trans_out = t
                # trans_out = trans_out * alpha + h[ntype] * (1 - alpha)
                # trans_out = t
                
                trans_out = trans_out + h[ntype]
                # trans_out = trans_out * (1 - alpha) + h[ntype] * alpha
                if self.use_norm:
                    new_h[ntype] = self.norms[n_id](trans_out)
                else:
                    new_h[ntype] = trans_out
            return new_h


class Mi(nn.Module):
    def __init__(self, node_dict, edge_dict, n_inp, n_hid, n_out, n_layers, n_heads, use_norm = True):
        super(Mi, self).__init__()
        self.node_dict = node_dict
        self.edge_dict = edge_dict
        self.gcs = nn.ModuleList()
        self.n_inp = n_inp
        self.n_hid = n_hid
        self.n_out = n_out
        self.n_layers = n_layers
        self.adapt_ws  = nn.ModuleList()
        for t in range(len(node_dict)):
            self.adapt_ws.append(nn.Linear(n_inp,   n_hid))
        for _ in range(n_layers):
            self.gcs.append(QTLayer(n_hid, n_hid, node_dict, edge_dict, n_heads, use_norm = use_norm))
        self.out = nn.Linear(n_hid, n_out)

    def forward(self, G,name_1,name_2,h):
        # h = {}
        # for ntype in G.ntypes:
        #     n_id = self.node_dict[ntype]
        #     h[ntype] = F.gelu(self.adapt_ws[n_id](h[ntype]))
        origin_h = h
        
        for i in range(self.n_layers):
            origin_h = self.gcs[i](G, origin_h)
        
        return self.out(origin_h[name_1]),self.out(origin_h[name_2])

class MineModel(torch.nn.Module):
    def __init__(self,n_heads,node_dict,edge_dict,q_e,t_e):
        super(MineModel, self).__init__()
        
        self.emb_size = 768
        self.need_num_heads = n_heads
        # self.q_emb = torch.nn.Parameter(torch.FloatTensor(q_e), requires_grad = True)
        # nn.init.xavier_uniform_(self.q_emb)
        self.q_emb = torch.nn.Parameter(torch.FloatTensor(q_e), requires_grad = False)

        
        self.t_emb = torch.nn.Parameter(torch.FloatTensor(t_e), requires_grad = False)
        self.q_len = self.q_emb.shape[0]
        self.in_size = self.q_emb.shape[1]
        self.mine = Mi(
            node_dict, edge_dict,
            n_inp=self.in_size,
            n_hid=self.emb_size,
            n_out=self.emb_size,
            n_layers=2,
            n_heads=4,
            use_norm = False)
            
        self.q_encoder = torch.nn.Sequential(
            torch.nn.Linear(self.in_size,self.emb_size*self.need_num_heads)
            )
        self.cluster_encoder = torch.nn.Linear(self.emb_size,10)
        self.num_clusters = 10
        self.cluster_centers = torch.nn.Parameter(torch.FloatTensor(self.num_clusters,self.emb_size), requires_grad = True)
        nn.init.xavier_uniform_(self.cluster_centers)
        self.cache = None
        self.graph = None
        # self.t_encoder = torch.nn.Linear(self.in_size,self.emb_size)
        # self.need_num_heads = need_num_heads
        



    def forward(self, G):
        if self.cache == None:
            sub_edge_col, sub_edge_row = G.edges(etype='give')
            tag_edge_1_col,tag_edge_1_row = G.edges(etype='inter_1')
            tag_edge_2_col,tag_edge_2_row = G.edges(etype='inter_2')
            
            add_row = torch.ones_like(sub_edge_row)
            q_len = self.q_emb.shape[0]
            add_edge_row = add_row * q_len
            for i in range(self.need_num_heads):
                if i == 0:
                    need_sub_edge_col = sub_edge_col
                    need_sub_edge_row = sub_edge_row
                else:
                    need_sub_edge_col = torch.cat((need_sub_edge_col,sub_edge_col),dim=0)

                    new_sub_edge_row = sub_edge_row + add_edge_row * i
                    need_sub_edge_row = torch.cat((need_sub_edge_row,new_sub_edge_row),dim=0)

            graph_data = { 
                ('tag', 'give', 'question'):(need_sub_edge_col,need_sub_edge_row),
                ('tag', 'inter_1', 'tag'):(tag_edge_1_col,tag_edge_1_row),
                ('tag', 'inter_2', 'tag'):(tag_edge_2_col,tag_edge_2_row)
            }
            self.graph = dgl.heterograph(graph_data)
            self.cache = 1
        


        h = {}
        
        #build graph and cache
        q_hs = self.q_encoder(self.q_emb)
        
        q_h_heads = torch.split(q_hs,self.emb_size,dim=1)
        q_h = torch.concat(q_h_heads,dim = 0)
        q_tile = torch.tile(self.q_emb, (self.need_num_heads,1))
        # q_h = q_tile
        
        q_h = q_h + q_tile

        # get weighted cluster centers
        q_cluster_assign = self.cluster_encoder(q_h) #shape(q_len*num_heads,num_clusters)
        q_cluster_assign = F.softmax(q_cluster_assign,dim=-1) #shape()
        weighted_cluster_centers = q_cluster_assign @ self.cluster_centers

        q_h = q_h + weighted_cluster_centers
        # t_h = self.t_encoder(self.t_emb)
        t_h = self.t_emb
        h['tag'] = t_h
        h['question']= q_h
        ques_heads,tags = self.mine(self.graph,'question','tag',h)


        ques_h = torch.split(ques_heads,self.q_len,dim=0) # shape()
        ques = torch.stack(ques_h,dim=1)# shape(q_len,num_heads,emb_size)
       
        return ques, tags