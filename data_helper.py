# -*- coding:utf-8 -*-
# from numpy.random import shuffle as np_shuffle
# from enum import _EnumDict

import numpy as np
import re
from collections import defaultdict
import random
import datetime
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score
import copy


class SignedGraph(object):
    def __init__(self, fpath, is_undiredted=True):
        self.G_pos = None
        self.G_neg = None
        self.edge_pos = None
        self.edge_neg = None

        self.node_all = set()
        self.read_signed_graph(fpath, is_undiredted)

    def read_signed_graph(self, fpath, is_undirected=False):
        '''

        :param fpath:
        :param is_undirected: add bidirectional edge
        :return: G_pos, G_neg
        '''
        G_pos = defaultdict(set)
        G_neg = defaultdict(set)
        edge_pos = []
        edge_neg = []
        node_all = set()

        with open(fpath) as f:
            for line in f:
                if len(line) > 0 and line[0] == '#':
                    continue
                else:
                    res = re.split(r'\s+', line.strip())
                    source_node, target_node, sign = map(int, res)

                    node_all.add(source_node)
                    node_all.add(target_node)

                    if sign == 1:
                        if target_node not in G_pos[source_node]:#remove repeated edge
                            G_pos[source_node].add(target_node)
                            edge_pos.append((source_node, target_node, sign))
                        if is_undirected:
                            if source_node not in G_pos[target_node]:
                                G_pos[target_node].add(source_node)
                                edge_pos.append((target_node, source_node, sign))

                    elif sign == -1:
                        if target_node not in G_neg[source_node]:
                            G_neg[source_node].add(target_node)
                            edge_neg.append((source_node, target_node, sign))
                        if is_undirected:
                            if source_node not in G_neg[target_node]:
                                G_neg[target_node].add(source_node)
                                edge_neg.append((target_node, source_node, sign))
                    else:
                        print('unknown sign:', sign)
                        exit(-1)

        self.G_pos = G_pos
        self.G_neg = G_neg
        self.edge_pos = edge_pos
        self.edge_neg = edge_neg
        self.node_all = node_all

        return G_pos, G_neg



def signet_read_edge_info(fpath):
    edge_list = []
    with open(fpath) as f:
        for line in f:
            if line[0] != '#':
                s, t, sign = map(int, re.split(r'\s+', line.strip()))
                edge_list.append([s, t, sign])

    return edge_list

def signet_save_edge_info(edge_list, fpath):
    with open(fpath, 'w') as f:
        for edge in edge_list:
            s = ' '.join(map(str, edge)) + '\n'
            f.write(s)

def BESIDE_trans_emb_to_xy(aux_parameter, edge_tuples, mode_choose):
    '''
    from emb and parameters to feature
    return [x,y]
    '''
    X = []
    Y = []
    epoch_emb, wi, wj, bedge, sta_w1_source, sta_w1_target, sta_b1_source, sta_b1_target, sta_w_for_score, sta_b_for_score,sta_w_for_score_combined, sta_b_for_score_combined  = aux_parameter

    for edge in edge_tuples:
        # e.g. [1,3,-1]
        y = edge[2] if edge[2] == 1 else 0  # 1->1,-1->0
        emb_1 = np.array(epoch_emb[edge[0]])
        emb_2 = np.array(epoch_emb[edge[1]])

        triad_emb_edge = np.matmul(emb_1, wi) + np.matmul(emb_2, wj) + bedge


        sta_source_fea_emb = np.matmul(emb_1, sta_w1_source) + sta_b1_source
        sta_target_fea_emb = np.matmul(emb_2, sta_w1_target) + sta_b1_target
        status_fea_vec = sta_source_fea_emb - sta_target_fea_emb

        if mode_choose == 'tri_sta':
            final_fea = np.zeros(shape=len(triad_emb_edge) + len(status_fea_vec))
            final_fea[:len(triad_emb_edge)] = triad_emb_edge
            final_fea[len(triad_emb_edge):len(triad_emb_edge) + len(status_fea_vec)] = status_fea_vec

        elif mode_choose == 'tri':
            final_fea = triad_emb_edge

        else:
            print('unknown mode_choose:{}', mode_choose)
            exit()

        X.append(final_fea)
        Y.append(y)
    return np.array(X), np.array(Y)


    
def BESIDE_check_link_prediction_task(dataset_train_fpath, dataset_test_fpath, sub_log_fpath, epoch_no, aux_parameter,
                                      mode_choose, extra_info=None):
    '''
    different mode_choose -> report performance of different tasks

    '''


    edge_method = '(xiWi+xjWj+b)'  # 'l2_weight'#'l1_weight'# 'hadamard' #


    edge_train = signet_read_edge_info(dataset_train_fpath)
    edge_test = signet_read_edge_info(dataset_test_fpath)


    if mode_choose == 'sta':  # compare directly
        epoch_emb, Wi, Wj, bedge, sta_w1_source, sta_w1_target, sta_b1_source, sta_b1_target, sta_w_for_score, sta_b_for_score,sta_w_for_score_combined, sta_b_for_score_combined = aux_parameter

        def get_source_score(node):
            tmp_emb = epoch_emb[node]
            return np.matmul(np.matmul(tmp_emb,sta_w1_source) + sta_b1_source,sta_w_for_score) + sta_b_for_score
        def get_target_score(node):
            tmp_emb = epoch_emb[node]
            return np.matmul(np.matmul(tmp_emb,sta_w1_target) + sta_b1_target,sta_w_for_score) + sta_b_for_score
        correct_num = 0
        total_num = len(edge_test)
        equal_num = 0
        for edge in edge_test:
            u, v, sign = edge
            sta_source_score_u = get_source_score(u)[0]
            sta_target_score_v = get_target_score(v)[0]
            if (sta_source_score_u < sta_target_score_v and sign == 1) or (sta_source_score_u > sta_target_score_v and sign == -1):
                correct_num += 1
            if sta_source_score_u == sta_target_score_v:
                equal_num += 1

        sta_cmp_acc = 1.0 * correct_num / total_num
        print(sub_log_fpath + '\n{}:test epoch {}: acc {:.4f}({}/{}|equ={}) ({})\n'.format(
                datetime.datetime.now().isoformat(), epoch_no, sta_cmp_acc,correct_num,total_num,equal_num ,edge_method))

        if extra_info:
            print('extra_info:', extra_info)
        print('writing result to', sub_log_fpath)

        with open(sub_log_fpath, 'a') as f:
            s = sub_log_fpath + '\n{}:test epoch {}: acc {:.4f} ({})\n'.format(
                datetime.datetime.now().isoformat(), epoch_no, sta_cmp_acc, edge_method)

            if extra_info:
                s += '{}\n'.format(extra_info)

            f.write(s)
    else: #tri and tri_sta

        edge_train_emb_X, edge_train_emb_Y = BESIDE_trans_emb_to_xy(aux_parameter, edge_train, mode_choose)
        edge_test_emb_X, edge_test_emb_Y = BESIDE_trans_emb_to_xy(aux_parameter, edge_test, mode_choose)


        #train LR and test
        lr = LogisticRegression()
        lr.fit(edge_train_emb_X, edge_train_emb_Y)

        test_y_score = lr.predict_proba(edge_test_emb_X)[:, 1]

        test_y_pred = lr.predict(edge_test_emb_X)

        lp_auc_score = roc_auc_score(edge_test_emb_Y, test_y_score, average='macro')

        lp_f1_score_macro = f1_score(edge_test_emb_Y, test_y_pred, average='macro')
        lp_f1_score_micro = f1_score(edge_test_emb_Y, test_y_pred, average='micro')
        lp_acc = accuracy_score(edge_test_emb_Y, test_y_pred)


        print('{}:test epoch {}: auc {:.4f} f1_macro {:.4f} f1_micro {:.4f} acc {:.4f} ({})\n'.format(
            datetime.datetime.now().isoformat(), epoch_no, lp_auc_score, lp_f1_score_macro, lp_f1_score_micro, lp_acc,
            edge_method))


        if extra_info:
            print('extra_info:', extra_info)
        print('writing result to', sub_log_fpath)

        with open(sub_log_fpath, 'a') as f:
            s = sub_log_fpath + '\n{}:test epoch {}: auc {:.4f} f1_macro {:.4f} f1_micro {:.4f} acc {:.4f} ({})\n'.format(
                datetime.datetime.now().isoformat(), epoch_no, lp_auc_score, lp_f1_score_macro, lp_f1_score_micro,
                lp_acc, edge_method)

            if extra_info:
                s += '{}\n'.format(extra_info)

            f.write(s)


def BESIDE_sta_gen_batch(batch_size, edge_train, node_train_set, G_pos_train_ori, G_neg_train_ori):
    '''
    sample for bridge edges

    :return: (i,j,true_sign_ij)
    '''

    G_pos_train = copy.deepcopy(G_pos_train_ori)
    G_neg_train = copy.deepcopy(G_neg_train_ori)

    node_train_list = list(node_train_set)
    np.random.shuffle(node_train_list)

    edge_train_list = list(edge_train)
    np.random.shuffle(edge_train_list)

    for node, neis in G_pos_train_ori.items():
        if node in neis:
            G_pos_train[node].remove(node)

    for node, neis in G_neg_train_ori.items():
        if node in neis:
            G_neg_train[node].remove(node)



    bridge_edge_num = 0

    G_pos_train_rev = defaultdict(set)
    G_neg_train_rev = defaultdict(set)
    for cur_node, neis in G_pos_train.items():
        for nei in neis:
            G_pos_train_rev[nei].add(cur_node)
    for cur_node, neis in G_neg_train.items():
        for nei in neis:
            G_neg_train_rev[nei].add(cur_node)

    def remove_self_loop(G_dict):
        G_dict_copy = copy.deepcopy(G_dict)
        for node,neis in G_dict_copy.items():
            if node in neis:
                G_dict[node].remove(node)
        return G_dict

    G_pos_train = remove_self_loop(G_pos_train)
    G_neg_train = remove_self_loop(G_neg_train)
    G_pos_train_rev = remove_self_loop(G_pos_train_rev)
    G_neg_train_rev = remove_self_loop(G_neg_train_rev)


    cur_batch = []
    for edge_idx, edge in enumerate(edge_train_list):

        i, j, sign = edge

        true_sign = 1 if sign == 1 else 0

        if i == j:
            continue

        bak_nodes_i = G_pos_train[i].union(G_neg_train[i]).union(G_pos_train_rev[i]).union(G_neg_train_rev[i])
        bak_nodes_j = G_pos_train[j].union(G_neg_train[j]).union(G_pos_train_rev[j]).union(G_neg_train_rev[j])
        bak_k_nodes = bak_nodes_i.intersection(bak_nodes_j)

        if not bak_k_nodes: # `bridge' edge
            bridge_edge_num += 1

            cur_batch.append([i, j, true_sign])
            if len(cur_batch) == batch_size:
                yield np.array(cur_batch)
                cur_batch = []

    if len(cur_batch) > 0:
        yield np.array(cur_batch)

    # print('status:bridge_edge_num:{}'.format(bridge_edge_num))




def BESIDE_tri_gen_batch(batch_size, edge_train, node_train_set, G_pos_train_ori, G_neg_train_ori,
                         max_one_edge_train_samples=16):
    '''
    sample for common edges (with triads)

    :return: (i,j,i,k,j,k,sign_ij,sign_ik,sign_jk)
    '''

    G_pos_train = copy.deepcopy(G_pos_train_ori)
    G_neg_train = copy.deepcopy(G_neg_train_ori)

    node_train_list = list(node_train_set)
    np.random.shuffle(node_train_list)

    edge_train_list = list(edge_train)
    np.random.shuffle(edge_train_list)



    for node, neis in G_pos_train_ori.items():
        if node in neis:
            G_pos_train[node].remove(node)

    for node, neis in G_neg_train_ori.items():
        if node in neis:
            G_neg_train[node].remove(node)

    G_pos_train_rev = defaultdict(set)
    G_neg_train_rev = defaultdict(set)
    for cur_node, neis in G_pos_train.items():
        for nei in neis:
            G_pos_train_rev[nei].add(cur_node)
    for cur_node, neis in G_neg_train.items():
        for nei in neis:
            G_neg_train_rev[nei].add(cur_node)

    def remove_self_loop(G_dict):
        G_dict_copy = copy.deepcopy(G_dict)
        for node,neis in G_dict_copy.items():
            if node in neis:
                G_dict[node].remove(node)
        return G_dict

    G_pos_train = remove_self_loop(G_pos_train)
    G_neg_train = remove_self_loop(G_neg_train)
    G_pos_train_rev = remove_self_loop(G_pos_train_rev)
    G_neg_train_rev = remove_self_loop(G_neg_train_rev)

    return_batch = []

    sampled_node_set = set()
    actual_train_node_set = set()


    for edge_idx, edge in enumerate(edge_train_list):
        actual_train_node_set.add(edge[0])
        actual_train_node_set.add(edge[1])

        cur_batch = []

        i, j, sign_ij = edge

        if sign_ij == -1:
            sign_ij = 0
        else:
            sign_ij = 1


        if i == j:
            continue

        bak_nodes_i = G_pos_train[i].union(G_neg_train[i]).union(G_pos_train_rev[i]).union(G_neg_train_rev[i])
        bak_nodes_j = G_pos_train[j].union(G_neg_train[j]).union(G_pos_train_rev[j]).union(G_neg_train_rev[j])
        bak_k_nodes = bak_nodes_i.intersection(bak_nodes_j)
        if not bak_k_nodes: # edges which do not have triads
            continue

        if len(bak_k_nodes) > max_one_edge_train_samples: # in case there are too many triads
            bak_k_nodes = random.sample(list(bak_k_nodes), max_one_edge_train_samples)
        else:
            bak_k_nodes = list(bak_k_nodes)

        for k in bak_k_nodes:
            tmp_ik = (i, k)
            tmp_jk = (j, k)
            sign_ik = 1
            sign_jk = 1
            if k in G_pos_train[i]:
                pass
            elif k in G_neg_train[i]:
                sign_ik = 0
            elif k in G_pos_train_rev[i]:
                tmp_ik = (k, i)
            elif k in G_neg_train_rev[i]:
                tmp_ik = (k, i)
                sign_jk = 0

            if k in G_pos_train[j]:
                pass
            elif k in G_neg_train[j]:
                sign_jk = 0
            elif k in G_pos_train_rev[j]:
                tmp_jk = (k, j)
            elif k in G_neg_train_rev[j]:
                tmp_jk = (k, j)
                sign_jk = 0
            cur_batch.append([i, j, tmp_ik[0], tmp_ik[1], tmp_jk[0], tmp_jk[1], sign_ij, sign_ik, sign_jk])

        for one_strip in cur_batch:
            sampled_node_set.add(one_strip[0])
            sampled_node_set.add(one_strip[1])
            if len(return_batch) == batch_size:
                yield np.array(return_batch)
                return_batch = []
            return_batch.append(one_strip)

    if len(return_batch) > 0:
        yield np.array(return_batch)



def signet_read_node_info(dataset_nodes_fpath):
    node_set = set(np.loadtxt(dataset_nodes_fpath, dtype=np.int32))
    return node_set

def signet_read_train_edge(dataset_train_fpath, dataset_nodes_fpath):
    edge_train = signet_read_edge_info(dataset_train_fpath)
    node_all_set = signet_read_node_info(dataset_nodes_fpath)

    G_pos_train = defaultdict(set)
    G_neg_train = defaultdict(set)
    node_train_set = set()

    for edge in edge_train:
        source_node, target_node, sign = edge
        if sign == 1:
            G_pos_train[source_node].add(target_node)
        elif sign == -1:
            G_neg_train[source_node].add(target_node)
        else:
            print('unknown sign:', sign)
            exit(-1)
        node_train_set.add(source_node)


    node_num = max(node_all_set) + 1  # in case the embedding lookup index out of range

    return node_train_set, G_pos_train, G_neg_train, node_num


def stacmp_read_train_edge(dataset_train_fpath, dataset_nodes_fpath):
    edge_train = signet_read_edge_info(dataset_train_fpath)
    node_all_set = signet_read_node_info(dataset_nodes_fpath)
    
    G_pos_train = defaultdict(set)
    G_neg_train = defaultdict(set)
    node_train_set = set()

    for edge in edge_train:
        source_node, target_node, sign = edge
        if sign == 1:
            G_pos_train[source_node].add(target_node)
        elif sign == -1:
            G_neg_train[source_node].add(target_node)
        else:
            print('unknown sign:', sign)
            exit(-1)
        node_train_set.add(source_node)

    node_num = max(node_all_set) + 1

    return node_train_set, G_pos_train, G_neg_train, node_num
