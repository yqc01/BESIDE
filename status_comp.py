# -*- coding:utf-8 -*-
import re
from collections import defaultdict
import sys
import os

import datetime
from data_helper import stacmp_read_train_edge,signet_read_edge_info

import pickle as pkl

import networkx as nx
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score
import numpy as np

'''
status comparison

'''
def feat_by_BESIDE_from_pkl(epoch_pkl_fpath, mode_choose):
    node_score = dict()
    
    with open(epoch_pkl_fpath, 'rb') as f:
        emb_dim, aux_parameter = pkl.load(f)

    epoch_emb, wi, wj, bedge, sta_w1_source, sta_w1_target, sta_b1_source, sta_b1_target, sta_w_for_score, sta_b_for_score,sta_w_for_score_combined, sta_b_for_score_combined  = aux_parameter

    sta_source_score_emb = []
    sta_target_score_emb = []
    for i in range(len(epoch_emb)):
        tmp_emb = epoch_emb[i]
        if mode_choose == 'BESIDE_sta':
            sta_source_score_emb.append( np.matmul(np.matmul(tmp_emb,sta_w1_source) + sta_b1_source,sta_w_for_score) + sta_b_for_score )
            sta_target_score_emb.append( np.matmul(np.matmul(tmp_emb,sta_w1_target) + sta_b1_target,sta_w_for_score) + sta_b_for_score )
        elif mode_choose == 'BESIDE_tri_sta':
            sss_sta = np.matmul(tmp_emb, sta_w1_source) + sta_b1_source
            sss_bal = np.matmul(tmp_emb,wi) + 0.5 * bedge
            
            
            sta_source_score_emb.append(
                np.matmul(np.concatenate([sss_sta,sss_bal],axis=0), sta_w_for_score_combined) + sta_b_for_score_combined)

            sts_sta = np.matmul(tmp_emb, sta_w1_target) + sta_b1_target
            sts_bal = np.matmul(tmp_emb, wi) + 0.5 * bedge

            sta_target_score_emb.append(
                np.matmul(np.concatenate([sts_sta,sts_bal],axis=0), sta_w_for_score_combined) + sta_b_for_score_combined)
        else:
            print('unknown mode_choose:{} in feat_by_BESIDE_from_pkl'.format(mode_choose))

    sta_source_score_emb = np.array(sta_source_score_emb)
    sta_target_score_emb = np.array(sta_target_score_emb)
    sta_source_score_emb = np.squeeze(sta_source_score_emb,axis=1)
    sta_target_score_emb = np.squeeze(sta_target_score_emb,axis=1)

    return sta_source_score_emb, sta_target_score_emb


def nx_pagerank(node_num,G_pos_train,alpha=0.85):
    #set to 0.85 following the setting in troll-trust
    edges = []
    for i in range(node_num):
        if i not in G_pos_train or len(G_pos_train[i]) == 0:
            edges.append((i,i))
        else:
            for nei in G_pos_train[i]:
                edges.append((i,nei))

    G = nx.DiGraph(edges)
    #print(G.number_of_nodes())
    pr = nx.pagerank(G, alpha=alpha)  
    
    return pr
    
def feat_by_pagerank(node_num,G_pos_train):
    G_emb = nx_pagerank(node_num,G_pos_train) 
    return G_emb

def judge_by_rank_score_for_BESIDE(rk_source_score,rk_target_score,edge_test):

    correct_num = 0.0
    total_num = len(edge_test)

    y_true = []
    y_pred = []

    equ0_cnt = 0

    for edge in edge_test:
        u,v,val = edge[0],edge[1],edge[2]

        if (rk_source_score[u] < rk_target_score[v] and val == 1) or (rk_source_score[u] > rk_target_score[v] and val == -1): 
            correct_num += 1
        if rk_source_score[u] < rk_target_score[v]:
            y_pred.append(1)
        elif rk_source_score[u] > rk_target_score[v]:
            y_pred.append(-1)
        else:
            equ0_cnt += 1 #== 0

            y_pred.append(1) #
        y_true.append(val)

    return correct_num / total_num



def judge_by_rank_score(rk_score,edge_test):
    correct_num = 0.0
    total_num = len(edge_test)

    
    for edge in edge_test:
        u,v,val = edge[0],edge[1],edge[2]
        
        if (rk_score[u] < rk_score[v] and val == 1) or (rk_score[u] > rk_score[v] and val == -1): 
            correct_num += 1
        
    return correct_num / total_num



def main(dataset_choose, method_choose, dataset_train_fpath, dataset_test_fpath, dataset_nodes_fpath, BESIDE_para_fpath):
    

    sub_log_fpath = 'log/{}_{}_status_cmp_log.txt'.format(dataset_choose,method_choose)
    print('write result to log:', sub_log_fpath)


    node_train_set,G_pos_train, G_neg_train, node_num = stacmp_read_train_edge(dataset_train_fpath, dataset_nodes_fpath)
    G_pos_train_rev = defaultdict(set)
    G_neg_train_rev = defaultdict(set)
    for node, neis in G_pos_train.items():
        for nei in neis:
            G_pos_train_rev[nei].add(node)
    for node, neis in G_neg_train.items():
        for nei in neis:
            G_neg_train_rev[nei].add(node)


    edge_test = signet_read_edge_info(dataset_test_fpath)


    # --- read rank score
    
    
    if 'BESIDE' in method_choose:
        
        epoch_pkl_fpath = BESIDE_para_fpath 
        BESIDE_sta_source_score_emb, BESIDE_sta_target_score_emb = feat_by_BESIDE_from_pkl(epoch_pkl_fpath, method_choose)

        acc = judge_by_rank_score_for_BESIDE(BESIDE_sta_source_score_emb, BESIDE_sta_target_score_emb, edge_test)
        
        with open(sub_log_fpath, 'a') as f:
            s = '{}: model: {}\n{}: acc: {:.6f}\n\n'.format(datetime.datetime.now(), BESIDE_para_fpath, method_choose, acc)
            f.write(s)
 
    elif method_choose == 'pagerank':
        rank_score_pagerank = feat_by_pagerank(node_num,G_pos_train)
        acc = judge_by_rank_score(rank_score_pagerank, edge_test)
        with open(sub_log_fpath, 'a') as f:
            s = '{}\n{}: acc: {:.6f}\n'.format(datetime.datetime.now(), method_choose, acc)
            f.write(s)
       
    else:
        print('unknown method')
        exit()



if __name__ == '__main__':
    usage_s = '''Usage: python status_comp.py <method_choose> <dataset_choose> <dataset_train_fpath> <dataset_test_fpath> <dataset_nodes_fpath> (<BESIDE_para_fpath>)
    
arguments:
method choose: select model method name ([BESIDE_tri_sta, BESIDE_sta, pagerank])
dataset_choose: select a name for your dataset (e.g. slashdot, epinions, wikirfa)
dataset_train_fpath: train file path (you can use preprocess_data to generate it), edgelist format
dataset_test_fpath: test file path (you can use preprocess_data to generate it), edgelist format
dataset_nodes_fpath: nodes id file path (you can use preprocess_data to generate it)
BESIDE_para_fpath: BESIDE model parameters file path

example:
        python status_comp.py BESIDE_tri_sta slashdot ./dataset/soc-sign-Slashdot090221.txt.map.train ./dataset/soc-sign-Slashdot090221.txt.map.test ./dataset/soc-sign-Slashdot090221.txt.map.nodes ./emb/example.emb.ep10
    '''
    if len(sys.argv) < 6:
        print(usage_s)
        exit()
    try:
        method_choose_list = ['BESIDE_tri_sta', 'BESIDE_sta', 'pagerank']
        method_choose = sys.argv[1]
        dataset_choose = sys.argv[2] #slashdot
    
        dataset_train_fpath = sys.argv[3] #r'./dataset/soc-sign-Slashdot090221.txt.map.train'
        dataset_test_fpath = sys.argv[4] #r'./dataset/soc-sign-Slashdot090221.txt.map.test'
        dataset_nodes_fpath = sys.argv[5] #r'./dataset/soc-sign-Slashdot090221.txt.map.nodes'
       
        BESIDE_para_fpath = ''
        if method_choose not in method_choose_list:
            print('unknown method')
            raise Exception
        elif method_choose == 'BESIDE_tri_sta' or method_choose == 'BESIDE_sta':
            BESIDE_para_fpath = sys.argv[6]    
       
    except Exception as e:
        print(e)
        print('arguments wrong format')
        print(usage_s)
        exit()
    
    main(dataset_choose, method_choose, dataset_train_fpath, dataset_test_fpath, dataset_nodes_fpath, BESIDE_para_fpath)

   
    

