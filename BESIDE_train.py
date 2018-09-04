# -*- coding:utf-8 -*-
#from collections import defaultdict
import tensorflow as tf
import numpy as np
import os
import time
import datetime
import sys

from data_helper import signet_read_train_edge, BESIDE_tri_gen_batch, BESIDE_sta_gen_batch, \
    BESIDE_check_link_prediction_task, signet_read_edge_info

from BESIDE_model import BESIDE

import pickle

np.random.seed(2017)

'''
BESIDE train (with test result -> log)
'''




def main(dataset_choose, mode_choose, emb_dim, epoch_num, dataset_train_fpath, dataset_test_fpath, dataset_nodes_fpath):
    # --- common arguments

    methodname = 'BESIDE_{}'.format(mode_choose)

    timestamp = time.strftime('%Y%m%d-%H%M%S', time.localtime())

    model_data_run_name = '{}_{}_{}'.format(timestamp, methodname, dataset_choose)

    chk_dirname = 'checkpoints'

    sub_log_fpath = 'log/{}.log'.format(model_data_run_name)
    out_emb_fpath = r'./emb/{}.emb'.format(model_data_run_name)


    print('write to log:{}'.format(sub_log_fpath))


    # --- model arguments
    batch_size = 32

    num_checkpoints = 15 





    if mode_choose == 'tri_sta':
        emb_dim = int(emb_dim / 2)

    reg_alpha = 1e-4

    learning_rate = 0.01


    # --- read data

    node_train_set, G_pos_train, G_neg_train, node_num = signet_read_train_edge(dataset_train_fpath,
                                                                                dataset_nodes_fpath)
    print('{}:read train_list over'.format(datetime.datetime.now()))

    edge_train = signet_read_edge_info(dataset_train_fpath)

    print('edge train number:{}'.format(len(edge_train)))



    with tf.Session() as sess:
        beside = BESIDE(node_num, emb_dim)
        global_step = tf.Variable(0, name='global_step', trainable=False)
        optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)

        # only triad part
        tri_grads_and_vars = optimizer.compute_gradients(beside.loss_only_tri)
        tri_train_op = optimizer.apply_gradients(tri_grads_and_vars, global_step=global_step)

        # only bridge(status) part
        sta_score_grads_and_vars = optimizer.compute_gradients(beside.loss_only_sta)
        sta_score_train_op = optimizer.apply_gradients(sta_score_grads_and_vars, global_step=global_step)

        # triad part + status part
        combined_bal_grads_and_vars = optimizer.compute_gradients(beside.loss_tri_combined)
        combined_bal_train_op = optimizer.apply_gradients(combined_bal_grads_and_vars, global_step=global_step)

        combined_sta_score_grads_and_vars = optimizer.compute_gradients(beside.loss_sta_combined)
        combined_sta_score_train_op = optimizer.apply_gradients(combined_sta_score_grads_and_vars,
                                                                global_step=global_step)

        #prepare model output dir
        saver = tf.train.Saver(tf.global_variables(), max_to_keep=num_checkpoints)


        sess.run(tf.global_variables_initializer())

        out_dir = os.path.abspath(os.path.join(os.path.curdir, "runs", model_data_run_name))
        print("Writing to {}\n".format(out_dir))
        checkpoint_dir = os.path.abspath(os.path.join(out_dir, chk_dirname))
        checkpoint_prefix = os.path.join(checkpoint_dir, "model")

        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)


        feed_dict = None
        step = 0



        for epoch in range(0, epoch_num):
            print('epoch {}:'.format(epoch))

            # just test its score
            # read parameters value
            epoch_emb, wi, wj, bedge, sta_w1_source, sta_w1_target, sta_b1_source, sta_b1_target, sta_w_for_score, sta_b_for_score, \
            sta_w_for_score_combined, sta_b_for_score_combined = sess.run(
                [beside.emb_w, beside.wi, beside.wj, beside.bedge, beside.sta_w1_source, beside.sta_w1_target,
                 beside.sta_b1_source, beside.sta_b1_target,
                 beside.sta_w_for_score, beside.sta_b_for_score, beside.sta_w_for_score_combined,
                 beside.sta_b_for_score_combined], feed_dict=feed_dict)


            extra_info_str = ''
            # write score to log

            aux_parameter = (epoch_emb, wi, wj, bedge, sta_w1_source, sta_w1_target, sta_b1_source, sta_b1_target,
                             sta_w_for_score, sta_b_for_score,sta_w_for_score_combined, sta_b_for_score_combined)
            BESIDE_check_link_prediction_task(dataset_train_fpath, dataset_test_fpath, sub_log_fpath, epoch, aux_parameter,
                                              mode_choose, extra_info_str)



            # save parameters
            if epoch > 0 and (epoch % 10 == 0):

                path = saver.save(sess, checkpoint_prefix, global_step=step)
                print("Saved model checkpoint to {}\n".format(path))
                with open(sub_log_fpath, 'a') as f:
                    timestr = datetime.datetime.now().isoformat()
                    s = '{}:save step:{}, epoch:{} ,path:{}\n'.format(timestr, step, epoch, path)
                    f.write(s)
                save_emb_info = [emb_dim, aux_parameter]  # T
                epoch_pkl_fpath = '{}.ep{}'.format(out_emb_fpath, epoch)
                print('writing to {}'.format(epoch_pkl_fpath))
                with open(epoch_pkl_fpath, 'wb') as f:
                    pickle.dump(save_emb_info, f)

            # ---- tri epoch
            if mode_choose != 'sta':  #
                print('{}:tri part start'.format(datetime.datetime.now().isoformat()))


                tri_batches = BESIDE_tri_gen_batch(batch_size, edge_train, node_train_set, G_pos_train,
                                                   G_neg_train, max_one_edge_train_samples=1)


                for idx, batch in enumerate(tri_batches):

                    input_bal_x_ijk = batch[:, 0:6]
                    input_sign_xijk = batch[:, 6:9]


                    feed_dict = {
                        beside.input_tri_x_ijk: np.array(input_bal_x_ijk),
                        beside.input_sign_xijk: np.array(input_sign_xijk),
                        beside.reg_alpha: reg_alpha

                    }

                    if mode_choose == 'tri':
                        _, loss, step = sess.run([tri_train_op, beside.loss_only_tri, global_step], feed_dict=feed_dict)
                    elif mode_choose == 'tri_sta':
                        _, loss, step = sess.run([combined_bal_train_op, beside.loss_tri_combined, global_step],
                                                 feed_dict=feed_dict)
                    else:
                        print('unknown mode_choose:{} in tri(tri_sta) process'.format(mode_choose))
                        exit()




            # ---sta
            if mode_choose != 'tri':
                print('{}:status part start'.format(datetime.datetime.now().isoformat()))


                sta_batches = BESIDE_sta_gen_batch(batch_size, edge_train, node_train_set, G_pos_train,
                                                   G_neg_train)

                for idx, batch in enumerate(sta_batches):

                    input_sta_x_i = np.expand_dims(batch[:, 0], axis=1)  #
                    input_sta_x_j = np.expand_dims(batch[:, 1], axis=1)  #

                    input_sta_true_sign_ij = np.expand_dims(batch[:, 2], axis=1)



                    feed_dict = {
                        beside.input_sta_x_i: np.array(input_sta_x_i),
                        beside.input_sta_x_j: np.array(input_sta_x_j),

                        beside.input_sta_true_sign_ij: np.array(input_sta_true_sign_ij),
                        beside.reg_alpha: reg_alpha

                    }


                    if mode_choose == 'sta':
                        _, loss, step = sess.run([sta_score_train_op, beside.loss_only_sta, global_step],
                                                 feed_dict=feed_dict)
                    elif mode_choose == 'tri_sta':
                        _, loss, step = sess.run([combined_sta_score_train_op, beside.loss_sta_combined, global_step],
                                                 feed_dict=feed_dict)
                    else:
                        print('unknown mode_choose:{} in sta(tri_sta) process'.format(mode_choose))
                        exit()




        #final epoch parameters
        epoch_emb, wi, wj, bedge, sta_w1_source, sta_w1_target, sta_b1_source, sta_b1_target, sta_w_for_score, sta_b_for_score, \
        sta_w_for_score_combined, sta_b_for_score_combined = sess.run(
            [beside.emb_w, beside.wi, beside.wj, beside.bedge, beside.sta_w1_source, beside.sta_w1_target,
             beside.sta_b1_source, beside.sta_b1_target,
             beside.sta_w_for_score, beside.sta_b_for_score, beside.sta_w_for_score_combined,
             beside.sta_b_for_score_combined], feed_dict=feed_dict)

        extra_info_str = ''


        aux_parameter = (epoch_emb, wi, wj, bedge, sta_w1_source, sta_w1_target, sta_b1_source, sta_b1_target,
                         sta_w_for_score, sta_b_for_score, sta_w_for_score_combined, sta_b_for_score_combined)
        BESIDE_check_link_prediction_task(dataset_train_fpath, dataset_test_fpath, sub_log_fpath, epoch_num, aux_parameter,
                                          mode_choose, extra_info_str)


    # get edge embedding and train/test

    epoch_pkl_fpath = '{}.ep{}'.format(out_emb_fpath, epoch_num)
    print('writing to {}'.format(epoch_pkl_fpath))
    save_emb_info = [emb_dim, aux_parameter]
    with open(epoch_pkl_fpath, 'wb') as f:
        pickle.dump(save_emb_info, f)


if __name__ == '__main__':

    if len(sys.argv) < 8:
        print('''Usage: python BESIDE_train.py <dataset_choose> <mode_choose> <emb_dim> <epoch_num> <dataset_train_fpath> <dataset_test_fpath> <dataset_nodes_fpath>
arguments:
dataset_choose: select a name for your dataset (e.g. slashdot, epinions, wikirfa)
mode_choose: three mode, [tri_sta, tri, sta]
emb_dim: embedding dimension for nodes
epoch_num: train epoch number
dataset_train_fpath: train file path (you can use preprocess_data to generate it), edgelist format
dataset_test_fpath: test file path (you can use preprocess_data to generate it), edgelist format
dataset_nodes_fpath: nodes id file path (you can use preprocess_data to generate it)

example:
        python BESIDE_train.py slashdot tri_sta 20 100 ./dataset/soc-sign-Slashdot090221.txt.map.train ./dataset/soc-sign-Slashdot090221.txt.map.test ./dataset/soc-sign-Slashdot090221.txt.map.nodes
''')
        exit()

    try:
        dataset_choose = sys.argv[1]
        mode_choose = sys.argv[2]  #all, tri, sta
        emb_dim = int(sys.argv[3])
        epoch_num = int(sys.argv[4])
        
        dataset_train_fpath = sys.argv[5] #r'./dataset/soc-sign-Slashdot090221.txt.map.train'
        dataset_test_fpath = sys.argv[6] #r'./dataset/soc-sign-Slashdot090221.txt.map.test'
        dataset_nodes_fpath = sys.argv[7] #r'./dataset/soc-sign-Slashdot090221.txt.map.nodes'
    except Exception as e:
        print(e)
        print('wrong arguments')
        exit()

    if mode_choose not in ['tri_sta','tri','sta']:
        print('wrong mode_choose: {} (all, tri, sta)'.format(mode_choose))
        exit()

    
    main(dataset_choose, mode_choose, emb_dim, epoch_num, dataset_train_fpath, dataset_test_fpath, dataset_nodes_fpath)

