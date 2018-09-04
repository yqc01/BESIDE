# -*- coding:utf-8 -*-
import numpy as np
import tensorflow as tf
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

'''
BESIDE model
'''


class BESIDE(object):
    def __init__(self, node_num, emb_dim):
        '''

        :param node_num:
        :param emb_dim:


        '''
        #--- triad input
        self.input_tri_x_ijk = tf.placeholder(tf.int32, shape=[None, 6], name='input_tri_x_ijk')

        self.reg_alpha = tf.placeholder(tf.float32, shape=None, name='reg_alpha')


        self.input_sign_xijk = tf.placeholder(tf.float32, shape=[None, 3], name='input_sign_xij')

        # --- bridge(status) input

        self.input_sta_x_i = tf.placeholder(tf.int32, shape=[None, 1], name='input_sta_x_i')
        self.input_sta_x_j = tf.placeholder(tf.int32, shape=[None, 1], name='input_sta_x_j')

        self.input_sta_true_sign_ij = tf.placeholder(tf.float32, shape=[None, 1], name='input_sta_true_sign_ij')

        # ---
        with tf.device('/cpu:0'), tf.name_scope('embedding'):

            self.emb_w = tf.Variable(initial_value=tf.zeros([node_num, emb_dim]), name='emb_w', dtype=tf.float32)
            self.emb_tri_x_ijk = tf.nn.embedding_lookup(self.emb_w, self.input_tri_x_ijk, name='emb_tri_x_ijk')

            self.emb_sta_x_i = tf.nn.embedding_lookup(self.emb_w,self.input_sta_x_i,name='emb_sta_x_i')
            self.emb_sta_x_j = tf.nn.embedding_lookup(self.emb_w,self.input_sta_x_j,name='emb_sta_x_j')

        minval_hid_layer_1 = -np.sqrt(6.0 / (emb_dim + emb_dim))
        self.wi = tf.Variable(initial_value=tf.random_uniform([emb_dim, emb_dim], minval=minval_hid_layer_1,
                                                              maxval=-minval_hid_layer_1), name='wi')
        self.wj = tf.Variable(initial_value=tf.random_uniform([emb_dim, emb_dim], minval=minval_hid_layer_1,
                                                              maxval=-minval_hid_layer_1), name='wj')
        self.bedge = tf.Variable(
            initial_value=tf.random_uniform([emb_dim], minval=minval_hid_layer_1, maxval=-minval_hid_layer_1),
            name='bedge')

        # --- only triad loss
        self.x_tri_list = tf.split(self.emb_tri_x_ijk, [1, 1, 1, 1, 1, 1], axis=1)  # x1~x6

        for i in range(6):
            self.x_tri_list[i] = tf.squeeze(self.x_tri_list[i], 1)

        self.sign_y_ij, self.sign_y_ik, self.sign_y_jk = tf.split(self.input_sign_xijk, [1, 1, 1],
                                                                  axis=1)

        self.tri_edge_ij = tf.matmul(self.x_tri_list[0], self.wi) + tf.matmul(self.x_tri_list[1], self.wj) + self.bedge
        self.tri_edge_ik = tf.matmul(self.x_tri_list[2], self.wi) + tf.matmul(self.x_tri_list[3], self.wj) + self.bedge
        self.tri_edge_jk = tf.matmul(self.x_tri_list[4], self.wi) + tf.matmul(self.x_tri_list[5], self.wj) + self.bedge


        minval_hid_layer_ii = -np.sqrt(6.0 / (emb_dim + 1))

        hid_w_2 = tf.Variable(
            initial_value=tf.random_uniform([emb_dim, 1], minval=minval_hid_layer_ii, maxval=-minval_hid_layer_ii),
            name='w2')
        hid_b_2 = tf.Variable(
            initial_value=tf.random_uniform([1], minval=minval_hid_layer_ii, maxval=-minval_hid_layer_ii), name='b2')


        final_ij = tf.matmul(self.tri_edge_ij, hid_w_2) + hid_b_2
        final_ik = tf.matmul(self.tri_edge_ik, hid_w_2) + hid_b_2
        final_jk = tf.matmul(self.tri_edge_jk, hid_w_2) + hid_b_2


        tri_regularizer = self.reg_alpha* tf.add_n([tf.nn.l2_loss(v) for v in [self.wi,self.wj,hid_w_2]])

        self.loss_only_tri = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=final_ij, labels=self.sign_y_ij) +
                                            tf.nn.sigmoid_cross_entropy_with_logits(logits=final_ik, labels=self.sign_y_ik) +
                                            tf.nn.sigmoid_cross_entropy_with_logits(logits=final_jk, labels=self.sign_y_jk))
        self.loss_only_tri += tri_regularizer


        # --- only bridge(status) loss


        self.sta_x_i, self.sta_x_j = self.emb_sta_x_i,self.emb_sta_x_j
        self.sta_x_i = tf.squeeze(self.sta_x_i,axis=1)
        self.sta_x_j = tf.squeeze(self.sta_x_j,axis=1)


        sta_layer1_size =  emb_dim
        sta_minval_hid_layer_1 = -np.sqrt(6.0 / (emb_dim + sta_layer1_size))
        self.sta_w1_source = tf.Variable(
            initial_value=tf.random_uniform([emb_dim, sta_layer1_size], minval=sta_minval_hid_layer_1,
                                            maxval=-sta_minval_hid_layer_1), name='sta_w1_in')
        self.sta_b1_source = tf.Variable(initial_value=tf.random_uniform([sta_layer1_size], minval=sta_minval_hid_layer_1,
                                                                         maxval=-sta_minval_hid_layer_1), name='sta_b1_in')
        self.sta_w1_target = tf.Variable(
            initial_value=tf.random_uniform([emb_dim, sta_layer1_size], minval=sta_minval_hid_layer_1,
                                            maxval=-sta_minval_hid_layer_1), name='sta_w1_out')
        self.sta_b1_target = tf.Variable(initial_value=tf.random_uniform([sta_layer1_size], minval=sta_minval_hid_layer_1,
                                                                         maxval=-sta_minval_hid_layer_1), name='sta_b1_out')


        sta_layer2_size = 1
        sta_minval_hid_layer_2 = -np.sqrt(6.0 / (sta_layer1_size + sta_layer2_size))


        self.sta_w_for_score = tf.Variable(
            initial_value=tf.random_uniform([sta_layer1_size, 1], minval=sta_minval_hid_layer_2,
                                            maxval=-sta_minval_hid_layer_2), name='sta_w_for_score')
        self.sta_b_for_score = tf.Variable(initial_value=tf.random_uniform([1], minval=sta_minval_hid_layer_2,
                                                                  maxval=-sta_minval_hid_layer_2), name='sta_b_for_score')



        self.sta_fea_i = tf.matmul(self.sta_x_i, self.sta_w1_source) + self.sta_b1_source
        self.sta_fea_j = tf.matmul(self.sta_x_j, self.sta_w1_target) + self.sta_b1_target



        self.score_i = tf.matmul(self.sta_fea_i,self.sta_w_for_score) + self.sta_b_for_score #source node score
        self.score_j = tf.matmul(self.sta_fea_j,self.sta_w_for_score) + self.sta_b_for_score #target node score


        self.sta_score_diff = - (self.score_i - self.score_j)
        self.loss_only_sta = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.sta_score_diff,
                                                                                    labels=self.input_sta_true_sign_ij
                                                                                    ))

        sta_score_regularizer = self.reg_alpha * tf.add_n([tf.nn.l2_loss(v) for v in [self.sta_w_for_score,self.sta_w1_source,self.sta_w1_target]])
        self.loss_only_sta = self.loss_only_sta + sta_score_regularizer



        #--- combined train loss (tri_sta_combined)
        #--- combined train loss for tri
        minval_hid_layer_ii_combined = -np.sqrt(6.0 / (emb_dim * 2 + 1))

        hid_w_2_combined = tf.Variable(
            initial_value=tf.random_uniform([emb_dim * 2, 1], minval=minval_hid_layer_ii_combined,
                                            maxval=-minval_hid_layer_ii_combined),
            name='w2_combined')
        hid_b_2_combined = tf.Variable(
            initial_value=tf.random_uniform([1], minval=minval_hid_layer_ii_combined,
                                            maxval=-minval_hid_layer_ii_combined), name='b2_combined')

        def get_sta_fea_diff(src_emb,tar_emb):
            src_fea = tf.matmul(src_emb, self.sta_w1_source) + self.sta_b1_source
            tar_fea = tf.matmul(tar_emb, self.sta_w1_target) + self.sta_b1_target

            return src_fea - tar_fea


        self.tri_sta_fea_diff_ij = get_sta_fea_diff(self.x_tri_list[0], self.x_tri_list[1])
        self.tri_sta_fea_diff_ik = get_sta_fea_diff(self.x_tri_list[2], self.x_tri_list[3])
        self.tri_sta_fea_diff_jk = get_sta_fea_diff(self.x_tri_list[4], self.x_tri_list[5])

        self.tri_edge_ij_combined = tf.concat([self.tri_edge_ij, self.tri_sta_fea_diff_ij], axis=1)
        self.tri_edge_ik_combined = tf.concat([self.tri_edge_ik, self.tri_sta_fea_diff_ik], axis=1)
        self.tri_edge_jk_combined = tf.concat([self.tri_edge_jk, self.tri_sta_fea_diff_jk], axis=1)


        final_ij_combined = tf.matmul(self.tri_edge_ij_combined, hid_w_2_combined) + hid_b_2_combined
        final_ik_combined = tf.matmul(self.tri_edge_ik_combined, hid_w_2_combined) + hid_b_2_combined
        final_jk_combined = tf.matmul(self.tri_edge_jk_combined, hid_w_2_combined) + hid_b_2_combined


        tri_regularizer_combined = self.reg_alpha * tf.add_n([tf.nn.l2_loss(v) for v in [self.wi, self.wj, hid_w_2,self.sta_w1_source, self.sta_w1_target]])


        self.loss_tri_combined = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(logits=final_ij_combined, labels=self.sign_y_ij) +
            tf.nn.sigmoid_cross_entropy_with_logits(logits=final_ik_combined, labels=self.sign_y_ik) +
            tf.nn.sigmoid_cross_entropy_with_logits(logits=final_jk_combined, labels=self.sign_y_jk))
        self.loss_tri_combined += tri_regularizer_combined


        #--- combined train loss for status
        sta_layer2_size = 1
        sta_layer1_size_combined = emb_dim * 2
        sta_minval_hid_layer_2_combined = -np.sqrt(6.0 / (sta_layer1_size_combined + sta_layer2_size))
        self.sta_w2_combined = tf.Variable(
            initial_value=tf.random_uniform([sta_layer1_size_combined, sta_layer2_size], minval=sta_minval_hid_layer_2_combined,
                                            maxval=-sta_minval_hid_layer_2_combined), name='sta_w2_combined')
        self.sta_b2_combined = tf.Variable(initial_value=tf.random_uniform([sta_layer2_size], minval=sta_minval_hid_layer_2_combined,
                                                                  maxval=-sta_minval_hid_layer_2_combined), name='sta_b2_combined')


        self.sta_w_for_score_combined = tf.Variable(
            initial_value=tf.random_uniform([sta_layer1_size_combined, 1], minval=sta_minval_hid_layer_2,
                                            maxval=-sta_minval_hid_layer_2), name='sta_w_for_score_combined')
        self.sta_b_for_score_combined = tf.Variable(initial_value=tf.random_uniform([1], minval=sta_minval_hid_layer_2,
                                                                           maxval=-sta_minval_hid_layer_2),
                                           name='sta_b_for_score_combined')


        self.sta_fea_i_combined = tf.matmul(self.sta_x_i, self.sta_w1_source) + self.sta_b1_source
        self.sta_fea_j_combined = tf.matmul(self.sta_x_j, self.sta_w1_target) + self.sta_b1_target

        self.tri_fea_i_combined = tf.matmul(self.sta_x_i, self.wi) + 0.5 * self.bedge
        self.tri_fea_j_combined = tf.matmul(self.sta_x_j, self.wj) + 0.5 * self.bedge

        self.sta_fea_i_combined = tf.concat([self.sta_fea_i_combined, self.tri_fea_i_combined], axis=1)
        self.sta_fea_j_combined = tf.concat([self.sta_fea_j_combined, self.tri_fea_j_combined], axis=1)


        self.score_i_combined = tf.matmul(self.sta_fea_i_combined,
                                 self.sta_w_for_score_combined) + self.sta_b_for_score_combined  # 这个就是降维后的sta_score,i是source node
        self.score_j_combined = tf.matmul(self.sta_fea_j_combined, self.sta_w_for_score_combined) + self.sta_b_for_score_combined  # 这个是target_node

        self.sta_score_diff_combined = - (self.score_i_combined - self.score_j_combined)  #
        self.loss_sta_combined = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.sta_score_diff_combined,
                                                                                        labels=self.input_sta_true_sign_ij
                                                                                        ))

        sta_score_regularizer_combined = self.reg_alpha * tf.add_n([tf.nn.l2_loss(v) for v in [self.sta_w_for_score_combined,self.sta_w1_source,self.sta_w1_target,self.wi,self.wj]])
        self.loss_sta_combined = self.loss_sta_combined + sta_score_regularizer_combined




