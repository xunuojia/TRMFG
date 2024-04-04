# coding=utf-8
from sklearn.model_selection import train_test_split

import numpy as np


import torch
from torch import nn
import dgl


import warnings
warnings.filterwarnings("ignore",category=DeprecationWarning)




def pretreatment(question_tags_list,question_embeddings,tag_name_embeddings):
    seen_index, unseen_index = train_test_split(np.arange(0, tag_name_embeddings.shape[0]), test_size=0.5, random_state=12345)
    
    q_t_edge = []
    q_t_edge_index_row = []
    q_t_edge_index_col = []
    for i, question_tags in enumerate(question_tags_list):
        for j in question_tags:
            q_t_edge.append([i, j])
            q_t_edge_index_row.append(i)
            q_t_edge_index_col.append(j)
    whole_num_questions = question_embeddings.shape[0]
    whole_num_tags = tag_name_embeddings.shape[0]
    q_t_edge_index = np.array(q_t_edge).T 
    seen_mask = np.zeros(whole_num_tags)
    seen_mask[seen_index] = 1
    
    seen_mask = seen_mask.astype(np.bool_)
    
    mask = seen_mask[q_t_edge_index_col]
    
    seen_q_t_edge_index = q_t_edge_index[:,mask]
    unseen_q_t_edge_index = q_t_edge_index[:,~mask]
    seen_q_t_edge = seen_q_t_edge_index.T
    seen_q_t_edge = seen_q_t_edge.tolist()
    unseen_q_t_edge = unseen_q_t_edge_index.T
    unseen_q_t_edge = unseen_q_t_edge.tolist()

    train_q_t_edge, test_q_t_edge = train_test_split(seen_q_t_edge, test_size=0.5, random_state=12345)
    train_q_t_dict = {}
    for i in train_q_t_edge:  
        train_q_t_dict.setdefault(i[0],[]).append(i[1])
    test_q_t_dict = {}
    for i in test_q_t_edge:
        train_q_t_dict.setdefault(i[0],[]).extend([])
        test_q_t_dict.setdefault(i[0],[]).append(i[1])
    for i in unseen_q_t_edge:
        train_q_t_dict.setdefault(i[0],[]).extend([])
        test_q_t_dict.setdefault(i[0],[]).append(i[1])
    
    return train_q_t_edge, train_q_t_dict, test_q_t_dict


def pretreatment_second(question_tags_list,question_embeddings,tag_name_embeddings):
    seen_index, unseen_index = train_test_split(np.arange(0, tag_name_embeddings.shape[0]), test_size=0.1, random_state=123)
    
    q_t_edge = []
    q_t_edge_index_row = []
    q_t_edge_index_col = []
    for i, question_tags in enumerate(question_tags_list):
        for j in question_tags:
            q_t_edge.append([i, j])
            q_t_edge_index_row.append(i)
            q_t_edge_index_col.append(j)
    whole_num_questions = question_embeddings.shape[0]
    whole_num_tags = tag_name_embeddings.shape[0]
    q_t_edge_index = np.array(q_t_edge).T 
    seen_mask = np.zeros(whole_num_tags)
    seen_mask[seen_index] = 1
    
    seen_mask = seen_mask.astype(np.bool_)
    
    mask = seen_mask[q_t_edge_index_col]
    
    seen_q_t_edge_index = q_t_edge_index[:,mask]
    unseen_q_t_edge_index = q_t_edge_index[:,~mask]
    seen_q_t_edge = seen_q_t_edge_index.T
    seen_q_t_edge = seen_q_t_edge.tolist()
    unseen_q_t_edge = unseen_q_t_edge_index.T
    unseen_q_t_edge = unseen_q_t_edge.tolist()

    train_q_t_edge = seen_q_t_edge
    train_q_t_dict = {}
    for i in train_q_t_edge:  
        train_q_t_dict.setdefault(i[0],[]).append(i[1])
    test_q_t_dict = {}
    # for i in test_q_t_edge:
    #     train_q_t_dict.setdefault(i[0],[]).extend([])
    #     test_q_t_dict.setdefault(i[0],[]).append(i[1])
    for i in unseen_q_t_edge:
        train_q_t_dict.setdefault(i[0],[]).extend([])
        test_q_t_dict.setdefault(i[0],[]).append(i[1])
    
    return train_q_t_edge, train_q_t_dict, test_q_t_dict#, seen_index,unseen_index





def build_graph_dict(question_tags_list,question_embeddings,tag_name_embeddings,tag_parent_child_edges,train_question_tag_edges):
    
    whole_num_questions = question_embeddings.shape[0]
    whole_num_tags = tag_name_embeddings.shape[0]

    whole_length = len(question_tags_list)

    tag_edge_index = np.array(tag_parent_child_edges).T 
    
    # question_embeddings = torch.tensor(question_embeddings)
    # tag_name_embeddings = torch.tensor(tag_name_embeddings)
    graph_q_t_edges = torch.tensor(train_question_tag_edges)
    tag_edge_index = torch.tensor(tag_edge_index)
    graph_q_t_edges_row = []
    graph_q_t_edges_col = []



    tag_edge_index_row,tag_edge_index_col = tag_edge_index[0],tag_edge_index[1]


    

    all_graph_q_t_edges = graph_q_t_edges
    for i,j in enumerate(all_graph_q_t_edges):
        graph_q_t_edges_row.append(j[0])
        graph_q_t_edges_col.append(j[1])

    graph_data = {
        # ('question', 'get', 'tag'):(graph_q_t_edges_row,graph_q_t_edges_col),
        ('tag', 'give', 'question'):(graph_q_t_edges_col,graph_q_t_edges_row),
        ('tag', 'inter_1', 'tag'):(tag_edge_index_row,tag_edge_index_col),
        ('tag', 'inter_2', 'tag'):(tag_edge_index_col,tag_edge_index_row)
    }


    G = dgl.heterograph(graph_data)


    node_dict = {}
    edge_dict = {}
    for ntype in G.ntypes:
        node_dict[ntype] = len(node_dict)
    for etype in G.etypes:
        edge_dict[etype] = len(edge_dict)
        G.edges[etype].data['id'] = torch.ones(G.number_of_edges(etype), dtype=torch.long) * edge_dict[etype] 

    
    return G, node_dict, edge_dict





