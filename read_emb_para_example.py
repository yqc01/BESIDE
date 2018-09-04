# -*- coding:utf-8 -*-
import re
from collections import defaultdict
import sys
import os
import pickle 

def main():
    epoch_pkl_fpath = './emb/example.emb.ep10'
    
    with open(epoch_pkl_fpath, 'rb') as f:
        [emb_dim, aux_parameter] = pickle.load(f)
        (epoch_emb, wi, wj, bedge, sta_w1_source, sta_w1_target, sta_b1_source, sta_b1_target,
                         sta_w_for_score, sta_b_for_score, sta_w_for_score_combined, sta_b_for_score_combined) = aux_parameter
        print(epoch_emb[:5])
        print(emb_dim)

        
if __name__ == '__main__':
    main()