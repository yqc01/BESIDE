#-*- coding:utf-8 -*-
from collections import defaultdict
import tensorflow as tf
import numpy as np
import os
import time
import datetime
import sys
import re


from data_helper import SignedGraph,signet_save_edge_info,signet_read_edge_info

np.random.seed(2017)

'''
preprocess original dataset
split train/test


'''

def preprocess_once(fpath_ori, fpath_out):
    '''
    map: index -> index + 1  (leave out index = 0 node, because sine need virtual node)

    '''

    G_pos = defaultdict(set)
    G_neg = defaultdict(set)
    edge_pos = []
    edge_neg = []

    G_has_out_link = set()

    with open(fpath_ori) as f, open(fpath_out, 'w') as fout:
        for line in f:
            if len(line) > 0 and line[0] == '#':

                fout.write(line)
            else:
                res = re.split(r'\s+', line.strip())
                source_node, target_node, sign = map(int, res)


                s = '{} {} {}\n'.format(source_node + 1, target_node + 1, sign)
                fout.write(s)


                if sign != 1 and sign != -1:
                    print('unknown sign:', sign)
                    exit(-1)



def main(dataset_choose,dataset_fpath_origin, dataset_fpath, dataset_train_fpath, dataset_test_fpath, dataset_nodes_fpath):
    # --- preprocess once

    is_undirected = False#True

    print('preprocess the dataset, writing to {}'.format(dataset_fpath))
    preprocess_once(dataset_fpath_origin, dataset_fpath)

    signedG = SignedGraph(dataset_fpath,is_undirected)#
    node_num = max(signedG.node_all) + 1  #in case embedding lookup out of range

    edge_all = np.array(signedG.edge_pos+signedG.edge_neg)#(source,target,sign)

    # --- split train/test
    shuffle_seed = 2017
    np.random.seed(shuffle_seed)
    shuffle_indices = np.random.permutation(np.arange(len(edge_all)))
    edge_all_shuffle = edge_all[shuffle_indices]

    test_sample_percentage = 0.2

    test_sample_index = -1 * int(test_sample_percentage * float(len(edge_all_shuffle)))
    edge_train, edge_test = edge_all_shuffle[:test_sample_index], edge_all_shuffle[test_sample_index:]

    print('edge train/test:{}/{}'.format(len(edge_train),len(edge_test)))

    #train
    signet_save_edge_info(edge_train, dataset_train_fpath)
    #test
    signet_save_edge_info(edge_test, dataset_test_fpath)

    #node id
    with open(dataset_nodes_fpath,'w') as f:
        for i in range(0,node_num):
            s = '{}\n'.format(i)
            f.write(s)



if __name__ == '__main__':
    if len(sys.argv) < 7:
        print('''Usage:python preprocess_data.py <dataset_choose> <dataset_fpath_origin> <dataset_fpath_output> <dataset_train_fpath> <dataset_test_fpath> <dataset_nodes_fpath>

arguments:
dataset_choose: select a name for your dataset (e.g. slashdot, epinions, wikirfa)
dataset_fpath_origin: input dataset file path
dataset_fpath_output: output dataset file path
dataset_train_fpath: output train file path
dataset_test_fpath: output test file path
dataset_nodes_fpath: output nodes id file path

example:
        python preprocess_data.py slashdot ./original_dataset/soc-sign-Slashdot090221.txt ./dataset/soc-sign-Slashdot090221.txt.map ./dataset/soc-sign-Slashdot090221.txt.map.train ./dataset/soc-sign-Slashdot090221.txt.map.test ./dataset/soc-sign-Slashdot090221.txt.map.nodes

''') 
        exit()

    dataset_choose = sys.argv[1] #'slashdot'

    dataset_fpath_origin = sys.argv[2]#r'./dataset/soc-sign-Slashdot090221.txt'

    dataset_fpath_output = sys.argv[3]#r'./dataset/soc-sign-Slashdot090221.txt.map' 

    dataset_train_fpath = sys.argv[4]#r'./dataset/soc-sign-Slashdot090221.txt.map.train'
    dataset_test_fpath = sys.argv[5]#r'./dataset/soc-sign-Slashdot090221.txt.map.test'
    dataset_nodes_fpath = sys.argv[6]#r'./dataset/soc-sign-Slashdot090221.txt.map.nodes'

    main(dataset_choose, dataset_fpath_origin, dataset_fpath_output, dataset_train_fpath, dataset_test_fpath, dataset_nodes_fpath)


