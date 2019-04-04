# -*- coding: utf-8 -*-

"""
Author: kingming

File: Alex_net.py

Time: 2018/12/28 下午8:20

License: (C) Copyright 2018, xxx Corporation Limited.

"""

import tensorflow as tf

'''
Input：图片尺寸224*224 
Conv1：卷积核11*11，步长4，96个filter（卷积核尺寸较大） 
ReLU 
LRN1 
Max pooling1：3*3，步长2 
Conv2：卷积核5*5，步长1，256个filter 
ReLU 
LRN2 
Max pooling2：3*3，步长2 
Conv3：卷积核3*3，步长1，384个filter 
ReLU 
Conv4：卷积核3*3，步长1，384个filter 
ReLU 
Conv5：卷积核3*3，步长1，256个filter 
ReLU 
Max pooling3：3*3，步长2 
FC1：4096 
ReLU 
FC2：4096 
ReLU 
FC3（Output）：1000

'''


class AlexNet(object):
    def __init__(self,length,width,num_class):
        self.length = length
        self.width = width
        self.num_class = num_class
        self.input_x = tf.placeholder(tf.float32,shape=[None,self.length,self.width,self.channel],name='input_x')
        self.input_y = tf.placeholder(tf.float32,shape = [None,self.num_class],name='input_y')
        self.keep_prob = tf.placeholder(tf.float32)

        self.logits = self.inference(self)
        self.loss = self.loss(self)
        self.correct_pred = self.correct_pred(self)
        self.accuracy = self.accuracy(self)
        self.pred = self.pred(self)

    def inference(self):
        # 第1个卷积层
        with tf.name_scope('conv1') as scope:
            kernel = tf.Variable(tf.truncated_normal([11, 11, 3, 64], dtype=tf.float32,stddev=1e-1), name='weights')
            conv = tf.nn.conv2d(self.input_x, kernel, [1, 4, 4, 1], padding='SAME')
            biases = tf.Variable(tf.constant(0.0, shape=[64], dtype=tf.float32),trainable=True, name='biases')
            bias = tf.nn.bias_add(conv, biases)
            conv1 = tf.nn.relu(bias, name=scope)


        # 添加LRN层和最大池化层
        lrn1 = tf.nn.lrn(conv1, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75, name='lrn1')
        pool1 = tf.nn.max_pool(lrn1,ksize = [1, 3, 3, 1],strides = [1, 2, 2, 1],padding = 'VALID',name='pool1')

        # 设计第2个卷积层
        with tf.name_scope('conv2') as scope:
            kernel = tf.Variable(tf.truncated_normal([5, 5, 64, 192], dtype=tf.float32,stddev = 1e-1), name = 'weights')
            conv = tf.nn.conv2d(pool1, kernel, [1, 1, 1, 1], padding='SAME')
            biases = tf.Variable(tf.constant(0.0, shape=[192], dtype=tf.float32),trainable = True, name = 'biases')
            bias = tf.nn.bias_add(conv, biases)
            conv2 = tf.nn.relu(bias, name=scope)



        # 对第2个卷积层的输出进行处理，同样也是先做LRN处理再做最大化池处理。
        lrn2 = tf.nn.lrn(conv2, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75, name='lrn2')
        pool2 = tf.nn.max_pool(lrn2,ksize=[1, 3, 3, 1],strides=[1, 2, 2, 1],padding='VALID',name='pool2')


        # 设计第3个卷积层
        with tf.name_scope('conv3') as scope:
            kernel = tf.Variable(tf.truncated_normal([3, 3, 192, 384],dtype=tf.float32,stddev=1e-1), name='weights')
            conv = tf.nn.conv2d(pool2, kernel, [1, 1, 1, 1], padding='SAME')
            biases = tf.Variable(tf.constant(0.0, shape=[384], dtype=tf.float32),trainable=True, name='biases')
            bias = tf.nn.bias_add(conv, biases)
            conv3 = tf.nn.relu(bias, name=scope)

        # 设计第4个卷积层
        with tf.name_scope('conv4') as scope:
            kernel = tf.Variable(tf.truncated_normal([3, 3, 384, 256],dtype=tf.float32,stddev=1e-1), name='weights')
            conv = tf.nn.conv2d(conv3, kernel, [1, 1, 1, 1], padding='SAME')
            biases = tf.Variable(tf.constant(0.0, shape=[256], dtype=tf.float32),trainable=True, name='biases')
            bias = tf.nn.bias_add(conv, biases)
            conv4 = tf.nn.relu(bias, name=scope)

        # 设计第5个卷积层
        with tf.name_scope('conv5') as scope:
            kernel = tf.Variable(tf.truncated_normal([3, 3, 256, 256],dtype=tf.float32,stddev=1e-1), name='weights')
            conv = tf.nn.conv2d(conv4, kernel, [1, 1, 1, 1], padding='SAME')
            biases = tf.Variable(tf.constant(0.0, shape=[256], dtype=tf.float32),trainable=True, name='biases')
            bias = tf.nn.bias_add(conv, biases)
            conv5 = tf.nn.relu(bias, name=scope)

        # 最大池化层
        pool5 = tf.nn.max_pool(conv5,ksize=[1, 3, 3, 1],strides=[1, 2, 2, 1],padding='VALID',name='pool5')

        # flattened6
        with tf.name_scope('flattened6') as scope:
            flattened = tf.reshape(pool5, shape=[-1, 6 * 6 * 256])

        # fc6
        with tf.name_scope('fc6') as scope:
            weights = tf.Variable(tf.truncated_normal([6 * 6 * 256, 4096],
                                                      dtype=tf.float32,
                                                      stddev=1e-1), name='weights')
            biases = tf.Variable(tf.constant(0.0, shape=[4096], dtype=tf.float32),
                                 trainable=True, name='biases')
            bias = tf.nn.xw_plus_b(flattened, weights, biases)
            fc6 = tf.nn.relu(bias)

        # dropout6
        with tf.name_scope('dropout6') as scope:
            dropout6 = tf.nn.dropout(fc6, self.keep_prob)

        # fc7
        with tf.name_scope('fc7') as scope:
            weights = tf.Variable(tf.truncated_normal([4096, 4096],
                                                      dtype=tf.float32,
                                                      stddev=1e-1), name='weights')
            biases = tf.Variable(tf.constant(0.0, shape=[4096], dtype=tf.float32),
                                 trainable=True, name='biases')
            bias = tf.nn.xw_plus_b(dropout6, weights, biases)
            fc7 = tf.nn.relu(bias)

        # dropout7
        with tf.name_scope('dropout7') as scope:
            dropout7 = tf.nn.dropout(fc7, self.keep_prob)

        # fc8
        with tf.name_scope('fc8') as scope:
            weights = tf.Variable(tf.truncated_normal([4096, self.num_classes],
                                                      dtype=tf.float32,
                                                      stddev=1e-1), name='weights')
            biases = tf.Variable(tf.constant(0.0, shape=[self.num_classes], dtype=tf.float32),
                                 trainable=True, name='biases')
            fc8 = tf.nn.xw_plus_b(dropout7, weights, biases)

        return fc8
    def loss(self):
        """
        Calculates loss with L2 regularization
        :param logits: output of softmax linear
        :param labels: a batch of labels, 1D tensor of size [batch_size]
        :return:
            total_loss: loss tensor
        """
        cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.logits,labels=self.input_y,
                                                                        name='cross_entropy_per_example')
        cross_entropy_mean = tf.reduce_mean(cross_entropy, name='cross_entropy')
        return cross_entropy_mean

    def train(self):
        train_step = tf.train.AdamOptimizer(1e-4).minimize(self.loss)
        return train_step
    def correct_pred(self):
        corr_pred = tf.equal(tf.arg_max(self.logits,1),tf.arg_max(self.input_y,1))
        return corr_pred
    def accuracy(self):
        accu = tf.reduce_mean(tf.cast(self.correct_pred,tf.float32))
        return accu

    def pred(self):
        predictions = tf.argmax(self.logits, axis=1, name="predictions")
        return predictions























