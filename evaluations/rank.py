# coding=utf-8

from tqdm import tqdm

import numpy as np
import torch

import tensorflow as tf


from grecx.vector_search.vector_search import VectorSearchEngine
from grecx.metrics.ranking import ndcg_score, precision_score, recall_score

import warnings
warnings.filterwarnings("ignore",category=DeprecationWarning)




        
def score(ground_truth, pred_items, k_list, metrics):
    pred_match = [1 if item in ground_truth else 0 for item in pred_items]

    max_k = k_list[-1]
    if len(ground_truth) > max_k:
        ndcg_gold = [1] * max_k
    else:
        ndcg_gold = [1] * len(ground_truth) + [0] * (max_k - len(ground_truth))

    res_score = []
    for metric in metrics:
        if metric == "ndcg":
            score_func = ndcg_score
        elif metric == "precision":
            score_func = precision_score
        elif metric == "recall":
            score_func = recall_score
        else:
            raise Exception("Not Found Metric : {}".format(metric))

        for k in k_list:
            if metric == "ndcg":
                res_score.append(score_func(ndcg_gold[:k], pred_match[:k]))
            else:
                res_score.append(score_func(ground_truth, pred_match[:k]))

    return res_score





def get_final_eval(user_items_dict, user_mask_items_dict, user_embedding, item_embedding, k_list=[10, 20], metrics=["ndcg"]):
    v_search = VectorSearchEngine(item_embedding)
    need_heads_num = user_embedding.shape[1]
    user_list = np.split(user_embedding,need_heads_num,axis=1)
    user_score_list = []
    user_pred_list = []
    for he in range(need_heads_num):
        users = user_list[he]
        if tf.is_tensor(users):
            users = users.numpy()
        else:
            users = np.asarray(users)


        user_emb = np.squeeze(users)

        user_indices = list(user_items_dict.keys())
        
        embedded_users = user_emb[user_indices]
        max_mask_items_length = max(len(user_mask_items_dict[user]) for user in user_indices)

        user_score, user_rank_pred_items = v_search.search(embedded_users, k_list[-1] + max_mask_items_length)
        # user_score_tensor = torch.from_numpy(user_score)
        # top_tensor = user_score_tensor.reshape(-1)
        # print(user_rank_pred_items)
        
        user_score_list.append(user_score)
        user_pred_list.append(user_rank_pred_items)

    # user_score_all = torch.concat(user_score_list,dim = 0)
    # top_value,top_indices=torch.topk(user_score_all,k_list[-1] + max_mask_items_length)
    max_mask_items_length = max(len(user_mask_items_dict[user]) for user in user_indices)
    need_len = k_list[-1] + max_mask_items_length
    # for i, indice in enumerate(top_indices):
    #     true_indice = indice%user_len
    #     true_j = indice//user_len
    
    temp_user_list =  user_score_list[0]
    user_len = temp_user_list.shape[0]
    
    for i in range(user_len):
        # test_list = []
        for kk in range(need_heads_num):
            save_kk = user_score_list[kk][i, :]
            if kk == 0:
                test_list = save_kk
            else:

                test_list = test_list + save_kk
        
        
        test_list = torch.tensor(test_list)
        top_value,top_indices = torch.topk(test_list,need_len)
        temp_list = []
        
        for j, indice in enumerate(top_indices):
            true_indice = indice%need_len
            # true_j = indice//need_len
            true_j = torch.div(indice, need_len,rounding_mode='floor')
            
            temp_list.extend([user_pred_list[true_j][i,true_indice]])
            
        temp_user_list[i] = temp_list
    # print(temp_user_list)
    res_scores = []
    for user, pred_items in tqdm(zip(user_indices, temp_user_list)):

        items = user_items_dict[user]
        mask_items = user_mask_items_dict[user]
        pred_items = [item for item in pred_items if item not in mask_items][:k_list[-1]]

        res_score = score(items, pred_items, k_list, metrics)

        res_scores.append(res_score)

    res_scores = np.asarray(res_scores)
    names = []
    for metric in metrics:
        for k in k_list:
            names.append("{}@{}".format(metric, k))

    # return list(zip(names, np.mean(res_scores, axis=0, keepdims=False)))
    return dict(zip(names, np.mean(res_scores, axis=0, keepdims=False)))





def evaluate_mean_global_metrics(user_items_dict, user_mask_items_dict,
                                 user_embedding, item_embedding,
                                 k_list=[10, 20], metrics=["ndcg"]):

    v_search = VectorSearchEngine(item_embedding)

    if tf.is_tensor(user_embedding):
        user_embedding = user_embedding.numpy()
    else:
        user_embedding = np.asarray(user_embedding)

    user_indices = list(user_items_dict.keys())
    embedded_users = user_embedding[user_indices]
    max_mask_items_length = max(len(user_mask_items_dict[user]) for user in user_indices)

    _, user_rank_pred_items = v_search.search(embedded_users, k_list[-1] + max_mask_items_length)

    res_scores = []
    for user, pred_items in tqdm(zip(user_indices, user_rank_pred_items)):

        items = user_items_dict[user]
        mask_items = user_mask_items_dict[user]
        pred_items = [item for item in pred_items if item not in mask_items][:k_list[-1]]

        res_score = score(items, pred_items, k_list, metrics)

        res_scores.append(res_score)

    res_scores = np.asarray(res_scores)
    names = []
    for metric in metrics:
        for k in k_list:
            names.append("{}@{}".format(metric, k))

    # return list(zip(names, np.mean(res_scores, axis=0, keepdims=False)))
    return dict(zip(names, np.mean(res_scores, axis=0, keepdims=False)))

    
def get_final_vision(user_items_dict, user_mask_items_dict,
                                 user_embedding, item_embedding,
                                 k_list=[10, 20], metrics=["ndcg"]):

    v_search = VectorSearchEngine(item_embedding)

    if tf.is_tensor(user_embedding):
        user_embedding = user_embedding.numpy()
    else:
        user_embedding = np.asarray(user_embedding)

    user_indices = list(user_items_dict.keys())
    embedded_users = user_embedding[user_indices]
    max_mask_items_length = max(len(user_mask_items_dict[user]) for user in user_indices)

    _, user_rank_pred_items = v_search.search(embedded_users, k_list[-1] + max_mask_items_length)

    record = []
    for user, pred_items in tqdm(zip(user_indices, user_rank_pred_items)):

        items = user_items_dict[user]
        mask_items = user_mask_items_dict[user]
        pred_items = [item for item in pred_items if item not in mask_items][:k_list[-1]]
        temp_dic = {"user":user, "true":items,"mask":mask_items,"pred":pred_items}
        record.append(temp_dic)



        

   
    return record


def get_final_mine_vision(user_items_dict, user_mask_items_dict,
                                 user_embedding, item_embedding,
                                 k_list=[10, 20], metrics=["ndcg"]):

    v_search = VectorSearchEngine(item_embedding)
    need_heads_num = user_embedding.shape[1]
    user_list = np.split(user_embedding,need_heads_num,axis=1)
    user_score_list = []
    user_pred_list = []
    for he in range(need_heads_num):
        users = user_list[he]
        if tf.is_tensor(users):
            users = users.numpy()
        else:
            users = np.asarray(users)


        user_emb = np.squeeze(users)

        user_indices = list(user_items_dict.keys())
        
        embedded_users = user_emb[user_indices]
        max_mask_items_length = max(len(user_mask_items_dict[user]) for user in user_indices)

        user_score, user_rank_pred_items = v_search.search(embedded_users, k_list[-1] + max_mask_items_length)
        # user_score_tensor = torch.from_numpy(user_score)
        # top_tensor = user_score_tensor.reshape(-1)
        # print(user_rank_pred_items)
        
        user_score_list.append(user_score)
        user_pred_list.append(user_rank_pred_items)

    # user_score_all = torch.concat(user_score_list,dim = 0)
    # top_value,top_indices=torch.topk(user_score_all,k_list[-1] + max_mask_items_length)
    max_mask_items_length = max(len(user_mask_items_dict[user]) for user in user_indices)
    need_len = k_list[-1] + max_mask_items_length
    # for i, indice in enumerate(top_indices):
    #     true_indice = indice%user_len
    #     true_j = indice//user_len
    
    temp_user_list =  user_score_list[0]
    user_len = temp_user_list.shape[0]
    
    for i in range(user_len):
        # test_list = []
        for kk in range(need_heads_num):
            save_kk = user_score_list[kk][i, :]
            if kk == 0:
                test_list = save_kk
            else:

                test_list = test_list + save_kk
        
        
        test_list = torch.tensor(test_list)
        top_value,top_indices = torch.topk(test_list,need_len)
        temp_list = []
        
        for j, indice in enumerate(top_indices):
            true_indice = indice%need_len
            # true_j = indice//need_len
            true_j = torch.div(indice, need_len,rounding_mode='floor')
            
            temp_list.extend([user_pred_list[true_j][i,true_indice]])
            
        temp_user_list[i] = temp_list
    # print(temp_user_list)
    record = []
    for user, pred_items in tqdm(zip(user_indices, user_rank_pred_items)):

        items = user_items_dict[user]
        mask_items = user_mask_items_dict[user]
        pred_items = [item for item in pred_items if item not in mask_items][:k_list[-1]]
        temp_dic = {"user":user, "true":items,"mask":mask_items,"pred":pred_items}
        record.append(temp_dic)



        

   
    return record