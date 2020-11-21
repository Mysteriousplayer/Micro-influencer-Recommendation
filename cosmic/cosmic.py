import numpy as np
import matplotlib.pyplot as plt
import argparse
import tensorflow as tf
import os
import time
from six.moves import xrange

vocab_size=420
batch_size = 32
input_size = 128
concept_layer1_size = 1024
pairs_size = 1
zzh = 0.001
epsilon = 1e-3
embed_size=100

def vector_inputs():
    image_placeholder = tf.placeholder(tf.float32, shape=(None,224,224,3))
    label_placeholder = tf.placeholder(tf.float32, shape=(None,vocab_size))
    return image_placeholder, label_placeholder

def get_batch(images_, labels_, step):
    if ((step + 1) * batch_size * pairs_size > len(images_)):
        images = images_[step * batch_size * pairs_size:]
        labels = labels_[step * batch_size * pairs_size:]
    else:
        images = images_[step * batch_size * pairs_size:(step + 1) * batch_size * pairs_size]
        labels = labels_[step * batch_size * pairs_size:(step + 1) * batch_size * pairs_size]
    return images, labels


def fill_feed_dict_train(images_train, labels_train, images_pl, labels_pl, step, keep_prob ,is_training):
    # Create the feed_dict for the placeholders filled with the next
    # `batch size` examples.
    images_feed, labels_feed = get_batch(
        images_train, labels_train,  step)
    feed_dict = {
        images: images_feed,
        labels: labels_feed,
        keep_prob: 0.5,
        is_training:True
    }
    return feed_dict


def fill_feed_dict_test(images_test, labels_test, images_pl, labels_pl, step, keep_prob, is_training):
    # Create the feed_dict for the placeholders filled with the next
    # `batch size` examples.
    images_feed, labels_feed = get_batch(
        images_test, labels_test, step)
    feed_dict = {
        images: images_feed,
        labels: labels_feed,
        keep_prob: 1,
        is_training:False
    }
    return feed_dict


def get_weights(shape, lambd):
    var = tf.Variable(tf.random_normal(shape, stddev=0.1))
    tf.add_to_collection('losses', tf.contrib.layers.l1_regularizer(lambd)(var))
    return var


def bn_layer(x, scope, is_training, epsilon=0.001, decay=0.99, reuse=None):
    """
    Performs a batch normalization layer
    Args:
        x: input tensor
        scope: scope name
        is_training: python boolean value
        epsilon: the variance epsilon - a small float number to avoid dividing by 0
        decay: the moving average decay
    Returns:
        The ops of a batch normalization layer
    """
    with tf.variable_scope(scope, reuse=reuse):
        shape = x.get_shape().as_list()
        # gamma: a trainable scale factor
        gamma = tf.get_variable("gamma", shape[-1], initializer=tf.constant_initializer(1.0), trainable=True)
        # beta: a trainable shift value
        beta = tf.get_variable("beta", shape[-1], initializer=tf.constant_initializer(0.0), trainable=True)
        moving_avg = tf.get_variable("moving_avg", shape[-1], initializer=tf.constant_initializer(0.0), trainable=False)
        moving_var = tf.get_variable("moving_var", shape[-1], initializer=tf.constant_initializer(1.0), trainable=False)
        if is_training:
            # tf.nn.moments == Calculate the mean and the variance of the tensor x
            avg, var = tf.nn.moments(x, np.arange(len(shape) - 1), keep_dims=True)
            avg = tf.reshape(avg, [avg.shape.as_list()[-1]])
            var = tf.reshape(var, [var.shape.as_list()[-1]])
            # update_moving_avg = moving_averages.assign_moving_average(moving_avg, avg, decay)
            update_moving_avg = tf.assign(moving_avg, moving_avg * decay + avg * (1 - decay))
            # update_moving_var = moving_averages.assign_moving_average(moving_var, var, decay)
            update_moving_var = tf.assign(moving_var, moving_var * decay + var * (1 - decay))
            control_inputs = [update_moving_avg, update_moving_var]
        else:
            avg = moving_avg
            var = moving_var
            control_inputs = []
        with tf.control_dependencies(control_inputs):
            output = tf.nn.batch_normalization(x, avg, var, offset=beta, scale=gamma, variance_epsilon=epsilon)

    return output


def bn_layer_top(x, scope, is_training, epsilon=0.001, decay=0.99):
    """
    Returns a batch normalization layer that automatically switch between train and test phases based on the
    tensor is_training
    Args:
        x: input tensor
        scope: scope name
        is_training: boolean tensor or variable
        epsilon: epsilon parameter - see batch_norm_layer
        decay: epsilon parameter - see batch_norm_layer
    Returns:
        The correct batch normalization layer based on the value of is_training
    """
    # assert isinstance(is_training, (ops.Tensor, variables.Variable)) and is_training.dtype == tf.bool

    return tf.cond(
        is_training,
        lambda: bn_layer(x=x, scope=scope, epsilon=epsilon, decay=decay, is_training=True, reuse=None),
        lambda: bn_layer(x=x, scope=scope, epsilon=epsilon, decay=decay, is_training=False, reuse=True),
    )


def conv2d(x, W, b, strides=1):
    # Conv2D wrapper, with bias and relu activation
    x = tf.nn.conv2d(x, W, strides=[1, strides, strides, 1], padding='SAME')
    x = tf.nn.bias_add(x, b)
    return tf.nn.relu(x)


def maxpool2d(x, k=2):
    # MaxPool2D wrapper
    return tf.nn.max_pool(x, ksize=[1, k, k, 1], strides=[1, k, k, 1],
                          padding='SAME')


graph1 = tf.Graph()
with graph1.as_default():
    keep_prob = tf.placeholder(tf.float32)
    keep_prob_2 = tf.placeholder(tf.float32)
    is_training = tf.placeholder(tf.bool, name='training')
    images, labels = vector_inputs()
    weights = {
        'wc1': tf.Variable(tf.random_normal([3, 3, 3, 32])),
        'wc2': tf.Variable(tf.random_normal([3, 3, 32, 64])),
        'wc3': tf.Variable(tf.random_normal([3, 3, 64, 64])),
        'wc4': tf.Variable(tf.random_normal([3, 3, 64, 128])),
        'wc5': tf.Variable(tf.random_normal([3, 3, 128, 128]))
    }

    biases = {
        'bc1': tf.Variable(tf.random_normal([32])),
        'bc2': tf.Variable(tf.random_normal([64])),
        'bc3': tf.Variable(tf.random_normal([64])),
        'bc4': tf.Variable(tf.random_normal([128])),
        'bc5': tf.Variable(tf.random_normal([128]))
    }
    # Convolution Layer
    conv1 = conv2d(images, weights['wc1'], biases['bc1'])
    conv1 = bn_layer_top(conv1,'bn1',is_training)
    conv1=tf.nn.relu(conv1)
    conv1 = maxpool2d(conv1, k=3)
    #------------------------------------
    conv2 = conv2d(conv1, weights['wc2'], biases['bc2'])
    conv2 = bn_layer_top(conv2,'bn2',is_training)
    conv2 = tf.nn.relu(conv2)
    conv3 = conv2d(conv2, weights['wc3'], biases['bc3'])
    conv3 = bn_layer_top(conv3,'bn3',is_training)
    conv3 = tf.nn.relu(conv3)
    conv3 = maxpool2d(conv3, k=2)
    #----------------------------------------
    conv4 = conv2d(conv3, weights['wc4'], biases['bc4'])
    conv4 = bn_layer_top(conv4,'bn4',is_training)
    conv4 = tf.nn.relu(conv4)
    conv5 = conv2d(conv4, weights['wc5'], biases['bc5'])
    conv5 = bn_layer_top(conv5,'bn5',is_training)
    conv5 = tf.nn.relu(conv5)
    conv5 = maxpool2d(conv5, k=2)

    #-----------------------------------------------
    fc0=tf.layers.Flatten()(conv5)
    w_fc1=get_weights([19*19*128, concept_layer1_size], zzh)
    dropout1 = tf.nn.dropout(w_fc1, keep_prob)
    b_fc1=tf.Variable(tf.random_normal([concept_layer1_size]))
    fc1=tf.matmul(fc0,dropout1)+b_fc1
    fc1=bn_layer_top(fc1,'bn6',is_training)
    fc1=tf.nn.leaky_relu(fc1, 0.01)
    #---------------------------------------------------
    w_fc2 = get_weights([1024, embed_size], zzh)
    fc2=tf.matmul(fc1, w_fc2)

    #----------------------------
    tables=[]
    table=[]
    index=0
    for i in range(0, 420):
        table.append(i)
    for j in range(0,batch_size):
        tables.append(table)

    look_up_table=tf.convert_to_tensor(tables)

    embedding_layer=tf.Variable(tf.random_uniform([vocab_size,embed_size],-1,1),name='embedding')
    embed=tf.nn.embedding_lookup(embedding_layer,look_up_table)
    #-----------------
    fc2_2=tf.reshape(fc2,(-1,1,embed_size))
    fc2_3=tf.tile(fc2_2,[1,vocab_size,1])

    inner_score=tf.multiply(embed,fc2_3)
    inner_score=tf.reduce_mean(inner_score,axis=-1)

    prediction = tf.nn.softmax(inner_score)
    pred = tf.argmax(prediction, 1)
    cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=inner_score, labels=labels))

    correct_pred = tf.equal(pred, tf.argmax(labels, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
    num=tf.to_float(tf.greater(labels,1e-16))
    num=tf.reduce_sum(num)
    accuracy_2=tf.reduce_sum(tf.multiply(prediction,labels))/(num+1e-16)

    LEARNING_RATE_BASE = 0.001
    LEARNING_RATE_DECAY = 0.99
    LEARNING_RATE_STEP = 100
    gloabl_steps = tf.Variable(0, trainable=False)
    learning_rate = tf.train.exponential_decay(LEARNING_RATE_BASE, gloabl_steps, LEARNING_RATE_STEP,
                                               LEARNING_RATE_DECAY, staircase=True)
    tf.add_to_collection('losses', cross_entropy)
    loss = tf.add_n(tf.get_collection('losses'))
    train_op = tf.train.AdamOptimizer(learning_rate).minimize(loss)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-m", "--model", required=True, help="path to output model")
    args = vars(ap.parse_args())
    Epoch_ = 25
    Step_ = 0
    Epoch = 0
    Part =6 # the number of training set parts
    Part2=2
    config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=True)
    config.gpu_options.per_process_gpu_memory_fraction = 0.8
    config.gpu_options.visible_device_list = "0"
    # 先引入dataset
    l_train_loss=[]
    l_test_loss=[]
    l_train_acc=[]
    l_test_acc = []
    is_train=False
    with tf.Session(graph=graph1, config=config) as sess:

        saver = tf.train.Saver(max_to_keep=15)

        init = tf.global_variables_initializer()
        sess.run(init)

        ckpt = tf.train.get_checkpoint_state(args["model"])
        if ckpt and ckpt.all_model_checkpoint_paths:
            path_ = ''
            for path in ckpt.all_model_checkpoint_paths:
                path_ = path
            print(path_)
            saver.restore(sess, path_)
        else:
            init = tf.global_variables_initializer()
            sess.run(init)

        for j in range(0, Epoch_):
            print('Epoch %d' % (Epoch))
            print('-----train-----')
            mean_loss = 0
            mean_accuracy = 0
            mean_accuracy2=0
            step_all=0
            for part in range(0,Part):
                print('loading...train')
                dir1='concept_train_data_'+str(part)+'_.npy'
                dir2 = 'concept_train_without_label_' + str(part) + '_.npy'
                trainX = np.load(dir1)
                trainY = np.load(dir2)
                print('ok')
                if(len(trainX)%(pairs_size*batch_size)==0):
                    Step_=len(trainX)/(pairs_size*batch_size)
                else:
                    Step_=int(len(trainX)/(pairs_size*batch_size))
                Step_=int(Step_)

                for step in xrange(Step_):
                    step_all+=1
                    start_time = time.time()
                    feed_dict = fill_feed_dict_train(trainX,trainY,images,labels,step,keep_prob,is_training)

                    _, loss_value ,accuracy_,is_train,accuracy2_= sess.run([train_op, loss,accuracy,is_training,accuracy_2],feed_dict=feed_dict)
                    mean_loss+=loss_value
                    mean_accuracy+=accuracy_
                    mean_accuracy2+=accuracy2_
                    duration = time.time() - start_time

                    if (step % 100 == 0 and step!=0):

                        print('Step %d: loss = %.2f (%.3f sec)' % (step, mean_loss/(step_all), duration))
                        print('acc:',mean_accuracy/step_all)
                        print('acc2:',mean_accuracy2/step_all)
            l_train_loss.append(mean_loss/(step_all))
            l_train_acc.append(mean_accuracy2/step_all)
            print('is_training:',is_train)
            globalstep=Epoch

            checkpoint_file = os.path.join(args["model"], 'model.ckpt')
            saver.save(sess, checkpoint_file, global_step=globalstep)


            print('-----test-----')
            test_mean_loss = 0.0
            test_mean_accuracy = 0.0
            test_mean_accuracy2=0.0
            step_all_2=0
            for part2 in range(0, Part2):
                print('loading...test')
                dir1_ = 'concept_test_data_' + str(part2) + '_.npy'
                dir2_ = 'concept_test_without_label_' + str(part2) + '_.npy'
                testX = np.load(dir1_)
                testY = np.load(dir2_)
                print('ok')
                if(len(testX)%(pairs_size*batch_size)==0):
                    tStep_=len(testX)/(pairs_size*batch_size)
                else:
                    tStep_=int(len(testX)/(pairs_size*batch_size))
                tStep_=int(tStep_)

                for t in xrange(tStep_):
                    step_all_2+=1
                    feed_dict = fill_feed_dict_test(testX,testY,images,labels,t,keep_prob,is_training)
                    test_loss,test_accuracy_,is_train,test_accuracy2_=sess.run([loss,accuracy,is_training,accuracy_2],feed_dict=feed_dict)
                    test_mean_loss+=test_loss
                    test_mean_accuracy+=test_accuracy_
                    test_mean_accuracy2+=test_accuracy2_
                    if(t % 100 == 0 and t!=0):
                        print('Step %d: loss = %.2f ' % (t, test_mean_loss/step_all_2))
                        print('acc:',test_mean_accuracy/step_all_2)
                        print('acc2:',test_mean_accuracy2/step_all_2)
            l_test_loss.append(test_mean_loss/step_all_2)
            l_test_acc.append(test_mean_accuracy2/step_all_2)
            print('is_training:',is_train)
            Epoch += 1

    plt.style.use("ggplot")
    plt.figure()
    N = Epoch_
    x_=np.arange(0,N)
    yy=np.asarray(l_train_loss)
    print(yy.shape)
    plt.plot(x_, np.asarray(l_train_loss), label="train_loss")
    plt.plot(x_, np.asarray(l_test_loss), label="val_loss")
    plt.title("Training/Validation Loss")
    plt.xlabel("Epoch #")
    plt.ylabel("Loss")
    plt.legend(loc="upper right")
    plt.savefig("./model_100_without/Loss plot.png")

    plt.style.use("ggplot")
    plt.figure()
    N = Epoch_
    plt.plot(x_, np.asarray(l_train_acc), label="train_acc")
    plt.plot(x_, np.asarray(l_test_acc), label="val_acc")
    plt.title("Training/Validation Acc")
    plt.xlabel("Epoch #")
    plt.ylabel("Accuracy")
    plt.legend(loc="upper right")
    plt.savefig("./model_100_without/Acc plot.png")
if __name__ == '__main__':
    main()
