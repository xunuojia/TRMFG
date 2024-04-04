# coding=utf-8
from collections import Counter
import itertools
from itertools import chain

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from tqdm import tqdm
import pickle
import numpy as np
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1,2"
import math 
import argparse
import csv

from layers import GCNModel,GATModel,APPNPModel,LightModel,PROFIT
import torch
from torch import nn
import torch_geometric
from torch_geometric.nn import GCNConv,APPNP,GATConv
from torch_geometric.data import Data
from utils import pretreatment,pretreatment_second

# from dgl.nn import HeteroEmbedding
import tensorflow as tf
from evaluations import get_final_eval, evaluate_mean_global_metrics
import warnings
warnings.filterwarnings("ignore",category=DeprecationWarning)

with open("data/question_tags.p", "rb") as f:
    question_tags_list = pickle.load(f)

with open("data/tag_parent_child_edges.p", "rb") as f:
    tag_parent_child_edges = pickle.load(f)

with open("data/question_embeddings.p", "rb") as f:
    question_embeddings = pickle.load(f)

with open("data/tag_name_embeddings.p", "rb") as f:
    tag_name_embeddings = pickle.load(f)

# with open("data/tag_intro_embeddings.p", "rb") as f:
#     tag_intro_embeddings = pickle.load(f)




whole_num_questions = question_embeddings.shape[0]
whole_num_tags = tag_name_embeddings.shape[0]


whole_length = len(question_tags_list)
# print(whole_length)


need_heads_num = 4

tag_edge_index = np.array(tag_parent_child_edges).T 



train_question_tag_edges, train_q_t_dict, test_q_t_dict = pretreatment_second(question_tags_list,question_embeddings,tag_name_embeddings)



tag_name_embeddings = torch.tensor(tag_name_embeddings)
tag_edge_index = torch.LongTensor(tag_edge_index)
# print(train_tag_name_embeddings.shape)
# print(tag_edge_index.shape)
# tag_adj = torch_geometric.utils.to_scipy_sparse_matrix(tag_edge_index)
tag_graph = Data(x=tag_name_embeddings, edge_index=tag_edge_index)










def info_bpr(a_embeddings, b_embeddings, pos_edges, reduction='mean'):

    if isinstance(pos_edges, list):
        pos_edges = np.array(pos_edges)

    # device = a_embeddings.device

    a_indices = pos_edges[:, 0]
    b_indices = pos_edges[:, 1]
    # print(a_indices)
    # print(b_indices)    

    if isinstance(pos_edges, torch.Tensor):
        num_pos_edges = pos_edges.size(0)
    else:
        num_pos_edges = len(pos_edges)


    num_b = b_embeddings.size(0)
    shape_b = b_indices.shape
    neg_b_indices = torch.randint(low=0, high=whole_num_tags, size=shape_b).to(device)

    embedded_a = a_embeddings[a_indices]
    embedded_b = b_embeddings[b_indices]
    embedded_neg_b = b_embeddings[neg_b_indices]
    

    # embedded_combined_b = torch.cat([embedded_b.unsqueeze(1), embedded_neg_b], 1)

    # logits = (embedded_combined_b @ embedded_a.unsqueeze(-1)).squeeze(-1)
    bce_criterion = nn.BCEWithLogitsLoss(weight = None, reduce = False)

    # info_bpr_loss = F.cross_entropy(logits, torch.zeros([num_pos_edges], dtype=torch.int64).to(device), reduction=reduction)
    pos_logits = torch.sum(embedded_a * embedded_b, axis=-1)
    neg_logits = torch.sum(embedded_a * embedded_neg_b, axis=-1)
    pos_label = torch.ones_like(pos_logits)
    neg_label = torch.zeros_like(neg_logits)
    # print(pos_logits.shape)
    # print(pos_label)
    pos_losses = bce_criterion(pos_logits, pos_label)
    neg_losses = bce_criterion(neg_logits, neg_label)

    
    mf_losses = pos_losses + neg_losses

    mf_loss = torch.mean(mf_losses)
    return mf_loss






question_embeddings = torch.tensor(question_embeddings)

device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
model = GCNModel(question_embeddings,768,768,cached=True,tag_drop_rate=0.5,que_drop_rate=0.3).to(device)
# model = GATModel(question_embeddings,768,192,4,tag_drop_rate=0.5,que_drop_rate=0.3).to(device)
# model = APPNPModel(question_embeddings,768,768,cached=True,k=2,alpha=0.1,tag_drop_rate=0.5,que_drop_rate=0.3).to(device)
# model = LightModel(question_embeddings,whole_num_tags,768,2,que_drop_rate=0.3).to(device)
# model = PROFIT(question_embeddings,768,768,cached=True,tag_drop_rate=0.5,que_drop_rate=0.3).to(device)
question_embeddings = torch.tensor(question_embeddings).to(device)
tag_graph = tag_graph.to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=2e-3)
lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.99)
max_recall_5 = 0.0
max_recall_10 = 0.0
max_recall_15 = 0.0
max_ndcg_5 = 0.0
max_ndcg_10 = 0.0
max_ndcg_15 = 0.0
for epoch in range(1, 1001):
    step_losses = []
    step_mf_loss_sum = 0.0
    step_l2_losses = []

    model.train()
    for step, batch_q_t_edges in enumerate(
            tf.data.Dataset.from_tensor_slices(train_question_tag_edges).shuffle(1000000).batch(4000)):
        # for step, (batch_user_item_edges,) in tqdm(enumerate(train_data_loader)):
        batch_q_t_edges = torch.tensor(batch_q_t_edges.numpy()).to(device)
        batch_q_t_edges = batch_q_t_edges.to(dtype = torch.long)
        question_h, tag_h = model(tag_graph)
        # question_h, tag_h = model(question_embeddings, tag_node, eindex)


        info_bpr_loss = info_bpr(question_h, tag_h, batch_q_t_edges)

        l2_loss = torch.tensor(0.0).to(device)
        for name, param in model.named_parameters():
            if "weight" in name or "embeddings" in name:
                # print(l2_loss.is_cuda,param.is_cuda)
                l2_loss += 0.5 * (param ** 2).sum()

        loss = info_bpr_loss + l2_loss * 1e-4

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        step_losses.append(loss.item())
        step_mf_loss_sum += info_bpr_loss * len(batch_q_t_edges)
        step_l2_losses.append(l2_loss.item())
    lr_scheduler.step()
    print("epoch = {}\tmean_loss = {}\tmean_mf_loss = {}\tmean_l2_loss = {}".format(epoch,
                                                                                    np.mean(step_losses),
                                                                                    step_mf_loss_sum / len(
                                                                                        train_question_tag_edges),
                                                                                    np.mean(step_l2_losses)
                                                                                    ))

    
    if epoch % 100 == 0:
        with torch.no_grad():
            model.eval()
            question_h, tag_emb = model(tag_graph)
            # question_h, tag_emb = model(question_embeddings,tag_node, eindex)
            # tag_h = tag_emb[test_index]
            tag_h = tag_emb
            
            mean_results_dict = evaluate_mean_global_metrics(test_q_t_dict, train_q_t_dict,
                                                                question_h.detach().cpu().numpy(),
                                                                tag_h.detach().cpu().numpy(),
                                                                k_list=[5,10,15,20], metrics=["precision", "recall", "ndcg"])
            print(mean_results_dict)

        recall_5 = mean_results_dict["recall@5"]
        recall_10 = mean_results_dict["recall@10"]
        recall_15 = mean_results_dict["recall@15"]
        ndcg_5 = mean_results_dict["ndcg@5"]
        ndcg_10 = mean_results_dict["ndcg@10"]
        ndcg_15 = mean_results_dict["ndcg@15"]
        if recall_5 > max_recall_5:
            max_recall_5 = recall_5
        if recall_10 > max_recall_10:
            max_recall_10 = recall_10
        if recall_15 > max_recall_15:
            max_recall_15 = recall_15
        if ndcg_5 > max_ndcg_5:
            max_ndcg_5 = ndcg_5
        if ndcg_10 > max_ndcg_10:
            max_ndcg_10 = ndcg_10
        if ndcg_15 > max_ndcg_15:
            max_ndcg_15 = ndcg_15
        print("max_recall@5: ", max_recall_5)
        print("max_recall@10: ", max_recall_10)
        print("max_recall@15: ", max_recall_15)
        print("max_ndcg@5: ", max_ndcg_5)
        print("max_ndcg@10: ", max_ndcg_10)
        print("max_ndcg@15: ", max_ndcg_15)

        if epoch % 1000 == 0:
        
            with open('0.9_gcn_see.csv', 'a+', encoding='UTF8', newline='') as f:
                writer = csv.writer(f)
                
        
                writer.writerow([max_recall_5])
                writer.writerow([max_recall_10])
                writer.writerow([max_recall_15])
                writer.writerow([max_ndcg_5])
                writer.writerow([max_ndcg_10])
                writer.writerow([max_ndcg_15])







print("HHHHHHHHHHHHHHHHHHHHHHHHHHHHHH")

