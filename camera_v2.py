import tensorflow as tf
import os
import xlrd
import time
from six.moves import xrange
import numpy as np
import xlsxwriter
from metrics_all import Metrics
#import matplotlib.pyplot as plt
import argparse
import random
batch_size = 8

focus_layer1_size = 1024
focus_layer2_size = 512
focus_layer3_size = 512
pairs_size = 2
zzh = 0.001
epsilon = 1e-3
margin1_score = 7.0
embed_size = 100
s_size=64

def load_trainset():
    print('loading...train')
    brands_historical_train = np.load('./brand_account_trainset_100_with_text_new.npy')
    ins_historical_train = np.load('./in_account_trainset_100_with_text_new.npy')
    brands_concept_train = np.load('./brand_concept_trainset_100_with_text_new.npy')
    ins_concept_train = np.load('./in_concept_trainset_100_with_text_new.npy')
    cor_score_train=np.load('./cor_score_trainset_v2.npy')
    influence_score_train = np.load('./influence_score_trainset_v2.npy')
    print('ok')
    return brands_historical_train, ins_historical_train, brands_concept_train, ins_concept_train, cor_score_train, influence_score_train


def load_validationset():
    print('loading...validation')
    brands_historical_test = np.load('./brand_account_testset_100_with_text_new.npy')
    ins_historical_test = np.load('./in_account_testset_100_with_text_new.npy')
    brands_concept_test = np.load('./brand_concept_testset_100_with_text_new.npy')
    ins_concept_test = np.load('./in_concept_testset_100_with_text_new.npy')
    cor_score_test = np.load('./cor_score_testset_v2.npy')
    influence_score_test = np.load('./influence_score_testset_v2.npy')
    print('ok')
    return brands_historical_test, ins_historical_test, brands_concept_test, ins_concept_test, cor_score_test, influence_score_test

def load_testset():
    print('loading...test')
    brands_historical_test2 = np.load('./brand_account_testset_2_100_with_text_new.npy')
    ins_historical_test2 = np.load('./in_account_testset_2_100_with_text_new.npy')
    brands_concept_test2 = np.load('./brand_concept_testset_2_100_with_text_new.npy')
    ins_concept_test2 = np.load('./in_concept_testset_2_100_with_text_new.npy')
    cor_score_test2 = np.load('./cor_score_testset_2_v2.npy')
    influence_score_test2 = np.load('./influence_score_testset_2_v2.npy')
    print('ok')
    return brands_historical_test2, ins_historical_test2, brands_concept_test2, ins_concept_test2, cor_score_test2, influence_score_test2

def save_recommendation_result(model_p,Ep,l_score,l_user,l_in,l_ist):
    file = './' + model_p + '/_' + str(Ep) + '.xlsx'
    workbook = xlsxwriter.Workbook(file)
    worksheet = workbook.add_worksheet(u'sheet1')
    index_n = 0
    for n in range(0, len(l_score)):
        worksheet.write(index_n, 0, l_user[index_n])
        worksheet.write(index_n, 1, l_in[index_n])
        worksheet.write(index_n, 2, l_ist[index_n])
        worksheet.write(index_n, 3, l_score[index_n])
        index_n += 1
    workbook.close()
def vector_inputs():
    brands_historical_placeholder = tf.placeholder(tf.float32, shape=(None, embed_size))
    ins_historical_placeholder = tf.placeholder(tf.float32, shape=(None, embed_size))
    brands_concept_placeholder = tf.placeholder(tf.float32, shape=(None, embed_size))
    ins_concept_placeholder = tf.placeholder(tf.float32, shape=(None, embed_size))
    cor_score_placeholder= tf.placeholder(tf.float32, shape=(None, 4))
    influence_score_placeholder=tf.placeholder(tf.float32, shape=(None, 4))
    return brands_historical_placeholder, ins_historical_placeholder, brands_concept_placeholder, ins_concept_placeholder,cor_score_placeholder,influence_score_placeholder


def get_batch(brands_historical_, ins_historical_, brands_concept_, ins_concept_, cor_score_, influence_score_, step):
    if ((step + 1) * batch_size * pairs_size > len(brands_historical_)):
        brands_historical = brands_historical_[step * batch_size * pairs_size:]
        ins_historical = ins_historical_[step * batch_size * pairs_size:]
        brands_concept = brands_concept_[step * batch_size * pairs_size * 10:]
        ins_concept = ins_concept_[step * batch_size * pairs_size * 10:]
        cor_score=cor_score_[step * batch_size:]
        influence_score=influence_score_[step * batch_size:]
    else:
        brands_historical = brands_historical_[step * batch_size * pairs_size:(step + 1) * batch_size * pairs_size]
        ins_historical = ins_historical_[step * batch_size * pairs_size:(step + 1) * batch_size * pairs_size]
        brands_concept = brands_concept_[step * batch_size * pairs_size * 10:(step + 1) * batch_size * pairs_size * 10]
        ins_concept = ins_concept_[step * batch_size * pairs_size * 10:(step + 1) * batch_size * pairs_size * 10]
        cor_score = cor_score_[step * batch_size :(step + 1) * batch_size]
        influence_score = influence_score_[step * batch_size :(step + 1) * batch_size]
    return brands_historical, ins_historical, brands_concept, ins_concept, cor_score, influence_score


def fill_feed_dict_train(brands_historical_train, ins_historical_train, brands_historical_pl, ins_historical_pl, brands_concept_train,
                         ins_concept_train,brands_concept_pl, ins_concept_pl, cor_score_train, influence_score_train,cor_score_pl, influence_score_pl,step, keep_prob, is_training):
    # Create the feed_dict for the placeholders filled with the next
    # `batch size` examples.
    brands_historical_feed, ins_historical_feed, brands_concept_feed, ins_concept_feed,cor_score_feed,influence_score_feed= \
        get_batch(brands_historical_train,ins_historical_train, brands_concept_train, ins_concept_train, cor_score_train, influence_score_train, step)
    feed_dict = {
        brands_historical: brands_historical_feed,
        ins_historical: ins_historical_feed,
        brands_concept: brands_concept_feed,
        ins_concept: ins_concept_feed,
        cor_score: cor_score_feed,
        influence_score: influence_score_feed,
        keep_prob: 0.6,
        is_training: True
    }
    return feed_dict


def fill_feed_dict_test(brands_historical_test, ins_historical_test, brands_historical_pl, ins_historical_pl, brands_concept_test,
                        ins_concept_test,brands_concept_pl, ins_concept_pl, cor_score_test, influence_score_test,
                        cor_score_pl, influence_score_pl,step, keep_prob, is_training):
    # Create the feed_dict for the placeholders filled with the next
    # `batch size` examples.
    brands_historical_feed, ins_historical_feed, brands_concept_feed, ins_concept_feed ,cor_score_feed,influence_score_feed= get_batch(
        brands_historical_test, ins_historical_test, brands_concept_test, ins_concept_test, cor_score_test, influence_score_test,step)
    feed_dict = {
        brands_historical: brands_historical_feed,
        ins_historical: ins_historical_feed,
        brands_concept: brands_concept_feed,
        ins_concept: ins_concept_feed,
        cor_score: cor_score_feed,
        influence_score: influence_score_feed,
        keep_prob: 1,
        is_training: False
    }
    return feed_dict


def get_weights(shape, lambd):
    var = tf.Variable(tf.random_normal(shape, stddev=0.1))
    tf.add_to_collection('losses', tf.contrib.layers.l1_regularizer(lambd)(var))
    return var

def func1(brands_historical,ins_historical,keep_prob):
    w_historical_1 = get_weights([embed_size, focus_layer1_size], zzh)
    dropout1 = tf.nn.dropout(w_historical_1, keep_prob)
    b_historical_1 = tf.Variable(tf.random_normal([focus_layer1_size], stddev=0.1))
    brands_historical_v1 = tf.nn.leaky_relu(tf.matmul(brands_historical, dropout1) + b_historical_1, 0.01)
    ins_historical_v1 = tf.nn.leaky_relu(tf.matmul(ins_historical, dropout1) + b_historical_1, 0.01)
    
    w_historical_2 = get_weights([focus_layer1_size, focus_layer2_size], zzh)
    dropout3 = tf.nn.dropout(w_historical_2, keep_prob)
    b_historical_2 = tf.Variable(tf.random_normal([focus_layer2_size], stddev=0.1))
    brands_historical_v2 = tf.nn.leaky_relu(tf.matmul(brands_historical_v1, dropout3) + b_historical_2, 0.01)
    ins_historical_v2 = tf.nn.leaky_relu(tf.matmul(ins_historical_v1, dropout3) + b_historical_2, 0.01)
    return brands_historical_v2, ins_historical_v2
    
def func2(brands_concept,ins_concept,keep_prob):
    w_concept_1 = get_weights([embed_size, focus_layer1_size], zzh)
    dropout2 = tf.nn.dropout(w_concept_1, keep_prob)
    b_concept_1 = tf.Variable(tf.random_normal([focus_layer1_size], stddev=0.1))
    brands_concept_v1 = tf.nn.leaky_relu(tf.matmul(brands_concept, dropout2) + b_concept_1, 0.01)
    ins_concept_v1 = tf.nn.leaky_relu(tf.matmul(ins_concept, dropout2) + b_concept_1, 0.01)
    
    w_concept_2 = get_weights([focus_layer1_size, focus_layer2_size], zzh)
    dropout4 = tf.nn.dropout(w_concept_2, keep_prob)
    b_concept_2 = tf.Variable(tf.random_normal([focus_layer2_size], stddev=0.1))
    brands_concept_v2 = tf.nn.leaky_relu(tf.matmul(brands_concept_v1, dropout4) + b_concept_2, 0.01)
    ins_concept_v2 = tf.nn.leaky_relu(tf.matmul(ins_concept_v1, dropout4) + b_concept_2, 0.01)
    return brands_concept_v2, ins_concept_v2

def func3(brand_embed,in_embed,keep_prob):
    w_embed_2 = get_weights([embed_size*2, 1024], zzh)
    b_embed_2 = tf.Variable(tf.random_normal([1024], stddev=0.1))
    dropout5 = tf.nn.dropout(w_embed_2, keep_prob)
    brand_embed_v2 = tf.nn.leaky_relu(tf.matmul(brand_embed, dropout5) + b_embed_2, 0.01)
    in_embed_v2 = tf.nn.leaky_relu(tf.matmul(in_embed, dropout5) + b_embed_2, 0.01)

    w_embed_3 = get_weights([1024, 1024], zzh)
    b_embed_3 = tf.Variable(tf.random_normal([1024], stddev=0.1))
    dropout6 = tf.nn.dropout(w_embed_3, keep_prob)
    brand_embed_v3 = tf.nn.leaky_relu(tf.matmul(brand_embed_v2, dropout6) + b_embed_3, 0.01)
    in_embed_v3 = tf.nn.leaky_relu(tf.matmul(in_embed_v2, dropout6) + b_embed_3, 0.01)

    w_embed_4 = get_weights([1024, 512], zzh)
    b_embed_4 = tf.Variable(tf.random_normal([512],stddev=0.1))
    brand_embed_v4 = tf.nn.leaky_relu(tf.matmul(brand_embed_v3, w_embed_4) + b_embed_4, 0.01)
    in_embed_v4 = tf.nn.leaky_relu(tf.matmul(in_embed_v3, w_embed_4) + b_embed_4, 0.01)
    return brand_embed_v4, in_embed_v4
def get_endorsement_score(cor_score):
    weight1 = tf.Variable(tf.random_normal([1,s_size], stddev=0.1))
    weight2 = tf.Variable(tf.random_normal([1,s_size], stddev=0.1))
    b1=tf.Variable(tf.random_normal([s_size], stddev=0.1))
    b2 = tf.Variable(tf.random_normal([s_size], stddev=0.1))
    weight1_v2 = tf.Variable(tf.random_normal([s_size,1], stddev=0.1))
    weight2_v2 = tf.Variable(tf.random_normal([s_size,1], stddev=0.1))
    
    p_cor1,p_cor2,n_cor1,n_cor2=tf.split(cor_score, [1, 1, 1, 1], 1)
    
    p_cor1=tf.reshape(p_cor1,[batch_size,1])
    p_cor2 = tf.reshape(p_cor2, [batch_size,1])
    n_cor1 = tf.reshape(n_cor1, [batch_size,1])
    n_cor2 = tf.reshape(n_cor2, [batch_size,1])
    
    
    p_cor1_v2=tf.sigmoid(tf.matmul(p_cor1,weight1)+b1)
    n_cor1_v2 = tf.sigmoid(tf.matmul(n_cor1, weight1) + b1)
    p_cor2_v2 = tf.sigmoid(tf.matmul(p_cor2, weight2) + b2)
    n_cor2_v2 = tf.sigmoid(tf.matmul(n_cor2, weight2) + b2)
    
    
    p_s1 = tf.sigmoid(tf.matmul(p_cor1_v2, weight1_v2)+tf.matmul(p_cor2_v2, weight2_v2))
    n_s1 = tf.sigmoid(tf.matmul(n_cor1_v2, weight1_v2)+tf.matmul(n_cor2_v2, weight2_v2))
    
    p_s1=tf.reshape(p_s1,[batch_size])
    n_s1 = tf.reshape(n_s1, [batch_size])
    
    s1=tf.concat([p_s1,n_s1],axis=0)
    s1=tf.reshape(s1, [-1,2])
    return s1
    
def get_influence_score(influence_score):
    weight3 = tf.Variable(tf.random_normal([1,s_size], stddev=0.1))
    weight4 = tf.Variable(tf.random_normal([1,s_size], stddev=0.1))
    b3 = tf.Variable(tf.random_normal([s_size], stddev=0.1))
    b4 = tf.Variable(tf.random_normal([s_size], stddev=0.1))
    weight3_v2 = tf.Variable(tf.random_normal([s_size,1], stddev=0.1))
    weight4_v2 = tf.Variable(tf.random_normal([s_size,1], stddev=0.1))
    
    p_in1, p_in2, n_in1, n_in2 = tf.split(influence_score, [1, 1, 1, 1], 1)
    
    p_in1 = tf.reshape(p_in1, [batch_size,1])
    p_in2 = tf.reshape(p_in2, [batch_size,1])
    n_in1 = tf.reshape(n_in1, [batch_size,1])
    n_in2 = tf.reshape(n_in2, [batch_size,1])
    
    p_in1_v2=tf.sigmoid(tf.matmul(p_in1, weight3) + b3)
    n_in1_v2 = tf.sigmoid(tf.matmul(n_in1, weight3) + b3)
    p_in2_v2 = tf.sigmoid(tf.matmul(p_in2, weight4) + b4)
    n_in2_v2 = tf.sigmoid(tf.matmul(n_in2, weight4) + b4)

    p_s2 = tf.sigmoid(tf.matmul(p_in1_v2, weight3_v2)+tf.matmul(p_in2_v2, weight4_v2))
    n_s2 = tf.sigmoid(tf.matmul(n_in1_v2, weight3_v2)+tf.matmul(n_in2_v2, weight4_v2))
    
    p_s2 = tf.reshape(p_s2, [batch_size])
    n_s2 = tf.reshape(n_s2, [batch_size])
    
    s2 = tf.concat([p_s2, n_s2], axis=0)
    s2 = tf.reshape(s2, [-1, 2])
    return s2

def set_seeds(seed):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    tf.set_random_seed(seed)
    np.random.seed(seed)
def set_global_determinism(seed):
    from tfdeterminism import patch
    patch()
    set_seeds(seed=seed)
    os.environ['TF_DETERMINISTIC_OPS']='1'
    os.environ['TF_CUDNN_DETERMINISTIC'] = '1'
    tf.config.threading.set_inter_op_parallelism_threads(1)
    tf.config.threading.set_intra_op_parallelism_threads(1)
SEED=42
graph1 = tf.Graph()
with graph1.as_default():
    set_global_determinism(seed=SEED)
    keep_prob = tf.placeholder(tf.float32)
    is_training = tf.placeholder(tf.bool, name='training')
    #historical: historical activities representation    concept: concept vector  
    #marketing: marketing direction representation
    #cor_score:  endorsement effect score   influence_score: influencer influence score
    brands_historical, ins_historical, brands_concept, ins_concept ,cor_score,influence_score= vector_inputs()
    # ----------------------------------------
    # function ψ
    brands_historical_v2, ins_historical_v2=func1(brands_historical,ins_historical,keep_prob)
    # function ρ
    brands_concept_v2, ins_concept_v2=func2(brands_concept,ins_concept,keep_prob)
    
    # BPAM
    brands_historical_v3 = tf.tile(brands_historical_v2, [1, 10])
    ins_historical_v3 = tf.tile(ins_historical_v2, [1, 10])
    brands_historical_v3 = tf.reshape(brands_historical_v3, [batch_size * pairs_size * 10, focus_layer2_size])
    ins_historical_v3 = tf.reshape(ins_historical_v3, [batch_size * pairs_size * 10, focus_layer2_size])

    attention_layer_1 = tf.multiply(brands_historical_v3, ins_concept_v2)
    attention_layer_2 = tf.multiply(ins_historical_v3, brands_concept_v2)

    marketing_weight1 = tf.reduce_mean(attention_layer_1, axis=1)
    marketing_weight2 = tf.reduce_mean(attention_layer_2, axis=1)

    marketing_weight1_output = tf.reshape(marketing_weight1, [batch_size * pairs_size, 10])
    marketing_weight2_output = tf.reshape(marketing_weight2, [batch_size * pairs_size, 10])

    marketing_weight1_output_v2 = tf.nn.softmax(marketing_weight1_output, -1)
    marketing_weight2_output_v2 = tf.nn.softmax(marketing_weight2_output, -1)

    marketing_weight1_softmax = tf.reshape(marketing_weight1_output_v2, [batch_size * pairs_size * 10, 1])
    marketing_weight2_softmax = tf.reshape(marketing_weight2_output_v2, [batch_size * pairs_size * 10, 1])
    # --------------------------------
    marketing_weight1_softmax_v2 = tf.tile(marketing_weight1_softmax, [1, embed_size])
    marketing_weight2_softmax_v2 = tf.tile(marketing_weight2_softmax, [1, embed_size])
    
    brand_marketing_embed = brands_concept * marketing_weight2_softmax_v2
    in_marketing_embed = ins_concept * marketing_weight1_softmax_v2

    brand_marketing_embed = tf.reshape(brand_marketing_embed, [batch_size * pairs_size, 10, embed_size])
    in_marketing_embed = tf.reshape(in_marketing_embed, [batch_size * pairs_size, 10, embed_size])

    brand_marketing_embed = tf.reduce_sum(brand_marketing_embed, axis=1)
    in_marketing_embed = tf.reduce_sum(in_marketing_embed, axis=1)

    brand_embed=tf.concat([brand_marketing_embed,brands_historical],axis=1)
    in_embed = tf.concat([in_marketing_embed,ins_historical],axis=1)
    #----------------------------------------------
    #function γ
    brand_embed_v4, in_embed_v4 = func3(brand_embed,in_embed,keep_prob)
    #----------------------------------------------

    a_b = tf.multiply(brand_embed_v4, in_embed_v4)
    a_b = tf.reduce_sum(a_b, axis=1)

    distance = a_b
    # ------------------
    distance = tf.reshape(distance, [batch_size, pairs_size])
    
    #endorsement_score
    s1=get_endorsement_score(cor_score)
    #influence_score
    s2=get_influence_score(influence_score)
    #ranking score
    score=tf.multiply(distance, s1)
    score = tf.multiply(score, s2)
    dist = tf.transpose(score)
    
    anchor_positive_dist, anchor_negative_dist = tf.split(dist, [1, 1], 0)
    anchor_positive = tf.reshape(anchor_positive_dist, [batch_size])
    anchor_negative = tf.reshape(anchor_negative_dist, [batch_size])

    margin1 = tf.ones([1, batch_size]) * margin1_score
    triplet_loss1 = anchor_negative_dist - anchor_positive_dist + margin1
    triplet_loss1 = tf.maximum(triplet_loss1, 0.0)
    valid_triplets1 = tf.to_float(tf.greater(triplet_loss1, 1e-16))
    num_positive_triplets1 = tf.reduce_sum(valid_triplets1)
    triplet_loss1 = tf.reduce_sum(triplet_loss1) / (num_positive_triplets1 + 1e-16)
    tf.add_to_collection('losses', triplet_loss1)
    loss = tf.add_n(tf.get_collection('losses'))

    LEARNING_RATE_BASE = 0.002
    LEARNING_RATE_DECAY = 0.99
    LEARNING_RATE_STEP = 24000
    G_steps = tf.Variable(0, trainable=False)
    learning_rate = tf.train.exponential_decay(LEARNING_RATE_BASE, G_steps, LEARNING_RATE_STEP,
                                               LEARNING_RATE_DECAY, staircase=True)
    #train_op = tf.train.AdamOptimizer(learning_rate).minimize(loss)
    train_op = tf.train.AdamOptimizer(learning_rate).minimize(loss, global_step=G_steps)

def main():
    ap = argparse.ArgumentParser()
    # ap.add_argument("-d", "--dataset", required=True, help="path to input dataset (i.e., directory of images)")
    ap.add_argument("-m", "--model", required=True, help="path to output model")
    ap.add_argument("-c", '--cuda', default=0, type=int, help='cuda')
    args = vars(ap.parse_args())
    
    Epoch_ = 40
    Step_ = 0
    Epoch = 0
    Part = 1  # the number of training set parts
    
    config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=True)
    config.gpu_options.per_process_gpu_memory_fraction = 0.9
    config.gpu_options.visible_device_list = str(args["cuda"])
    # metrics
    l_train_loss = []
    l_test_loss = []
    l_r_10 = []
    l_r_50 = []
    l_medr = []
    l_auc = []
    l_cauc = []
    l_mrr=[]
    l_map=[]
    with tf.Session(graph=graph1, config=config) as sess:

        init = tf.global_variables_initializer()
        sess.run(init)
        saver = tf.train.Saver(max_to_keep=15)
        #------------------------------loading data------------------------------------------------
        brands_historical_train, ins_historical_train, brands_concept_train, ins_concept_train, cor_score_train, influence_score_train = load_trainset()
        brands_historical_test, ins_historical_test, brands_concept_test, ins_concept_test, cor_score_test, influence_score_test = load_validationset()
        brands_historical_test2, ins_historical_test2, brands_concept_test2, ins_concept_test2, cor_score_test2, influence_score_test2 = load_testset()

        for j in range(0, Epoch_):

            print('--------------------------train---------------------------')
            print('Epoch %d' % (Epoch))
            ckpt = tf.train.get_checkpoint_state(args["model"])
            if ckpt and ckpt.all_model_checkpoint_paths:
                path_ = ''
                for path in ckpt.all_model_checkpoint_paths:
                    path_ = path
                print(path_)
                # saver = tf.train.Saver(tf.historical_variables())
                saver.restore(sess, path_)

            for part in range(0, Part):
                if (len(brands_historical_train) % (pairs_size * batch_size) == 0):
                    Step_ = len(brands_historical_train) / (pairs_size * batch_size)
                else:
                    Step_ = int(len(brands_historical_train) / (pairs_size * batch_size))
                Step_ = int(Step_)
                mean_loss = 0
                for step in xrange(Step_):
                    start_time = time.time()
                    feed_dict = fill_feed_dict_train(brands_historical_train, ins_historical_train, brands_historical, ins_historical,
                                                     brands_concept_train, ins_concept_train, brands_concept, ins_concept,
                                                     cor_score_train, influence_score_train, cor_score, influence_score, step, keep_prob,is_training)
                    _, loss_value, is_train = sess.run([train_op, loss, is_training], feed_dict=feed_dict)

                    mean_loss += loss_value
                    duration = time.time() - start_time
                    loss1 = sess.run(triplet_loss1, feed_dict=feed_dict)
                    if (step % 1000 == 0 and step != 0):
                        neg, pos, w1 = sess.run([anchor_positive, brands_historical_v2, marketing_weight1_output_v2],
                                                feed_dict=feed_dict)
                        #weight1_,weight2_,weight3_,weight4_=sess.run([p_cor1_v2,p_cor2_v2,p_in1_v2,p_in2_v2],
                                                #feed_dict=feed_dict)
                        lr, gs = sess.run([learning_rate, G_steps], feed_dict=feed_dict)
                        print(lr, gs)
                        #print(neg)
                        #print(pos)
                        #print('w1:', w1)
                        print('triplet:', loss1)
                        #print(weight4_)
                        print('Step %d: loss = %.2f (%.3f sec)' % (step, mean_loss / step, duration))
                    if (step == Step_):
                        l_train_loss.append(mean_loss / step)
            print('is_training:', is_train)
            globalstep = Epoch

            checkpoint_file = os.path.join(args["model"], 'model.ckpt')
            saver.save(sess, checkpoint_file, global_step=globalstep)

            print('---------------------validation-------------------------')
            # saver = tf.train.Saver(var_list)
            ckpt = tf.train.get_checkpoint_state(args["model"])
            if ckpt and ckpt.all_model_checkpoint_paths:
                path_ = ''
                for path in ckpt.all_model_checkpoint_paths:
                    path_ = path
                print(path_)
                saver.restore(sess, path_)

            if (len(brands_historical_test) % (pairs_size * batch_size) == 0):
                tStep_ = len(brands_historical_test) / (pairs_size * batch_size)
            else:
                tStep_ = int(len(brands_historical_test) / (pairs_size * batch_size))
            tStep_ = int(tStep_)
            test_mean_loss = 0.0

            for t in xrange(tStep_):

                feed_dict = fill_feed_dict_test(brands_historical_test, ins_historical_test, brands_historical, ins_historical,
                                                brands_concept_test, ins_concept_test, brands_concept, ins_concept,
                                                cor_score_test,influence_score_test,cor_score,influence_score,t, keep_prob, is_training)
                test_loss, is_train = sess.run([loss, is_training], feed_dict=feed_dict)
                test_mean_loss += test_loss
                if (t % 500 == 0 and t != 0):
                    bbb, bbb2, w1 = sess.run([anchor_positive, brands_historical_v2, marketing_weight1_output_v2],
                                             feed_dict=feed_dict)
                    #print(bbb)
                    #print(bbb2)
                    #print('w1:', w1)
                    print('Step %d: loss = %.2f ' % (t, test_mean_loss / t))
                if (t == tStep_):
                    l_test_loss.append(test_mean_loss / t)
            print('is_training:', is_train)
            
            
            print('------------------test------------------------------')
            # load testset infor
            ExcelFile1 = xlrd.open_workbook('./testset_r.xlsx')
            sheet1 = ExcelFile1.sheet_by_index(0)
            l_user = []
            l_in = []
            l_ist = []
            l_score = []
            index = 0

            if (len(brands_historical_test2) % (pairs_size * batch_size) == 0):
                ttStep_ = len(brands_historical_test2) / (pairs_size * batch_size)
            else:
                ttStep_ = int(len(brands_historical_test2) / (pairs_size * batch_size))
            ttStep_ = int(ttStep_)
            test_mean_loss = 0.0

            for t in xrange(ttStep_):

                feed_dict = fill_feed_dict_test(brands_historical_test2, ins_historical_test2, brands_historical, ins_historical,
                                                brands_concept_test2, ins_concept_test2, brands_concept, ins_concept,
                                                cor_score_test2,influence_score_test2,cor_score,influence_score,t, keep_prob, is_training)
                dist1, test_loss, is_train = sess.run([anchor_positive, loss, is_training], feed_dict=feed_dict)
                test_mean_loss += test_loss

                for xyz in range(len(dist1)):
                    xx = dist1[xyz]
                    user = sheet1.cell(index, 0).value.encode('utf-8').decode('utf-8-sig')
                    influencer = sheet1.cell(index, 1).value.encode('utf-8').decode('utf-8-sig')
                    ist = sheet1.cell(index, 2).value
                    l_user.append(user)
                    l_in.append(influencer)
                    l_ist.append(ist)
                    l_score.append(xx)
                    index += 1

            print('is_training:', is_train)
            
            #--------------Measurement-----------------------------------
            medr, r10, r50 = Metrics.metrics(l_user, l_in, l_ist, l_score)
            auc, cauc = Metrics.auc(l_user, l_in, l_ist, l_score)
            mrr=Metrics.mrr(l_user, l_in, l_ist, l_score)
            map=Metrics.map(l_user, l_in, l_ist, l_score)
            
            l_medr.append(medr)
            l_r_10.append(r10)
            l_r_50.append(r50)
            l_auc.append(auc)
            l_cauc.append(cauc)
            l_mrr.append(mrr)
            l_map.append(map)
            
            save_recommendation_result(args["model"],Epoch,l_score,l_user,l_in,l_ist)
            
            print(len(l_user))
            Epoch += 1
            for ii in range(0, Epoch):
                print(ii)
                print(l_r_10[ii], l_r_50[ii], l_medr[ii], l_auc[ii], l_cauc[ii],l_mrr[ii],l_map[ii])


if __name__ == '__main__':
    main()
