import tensorflow as tf
import tensorflow.contrib as contrib
import numpy as np
import scipy.io as sio
from PIL import Image


def conv(inputs, num_out, ksize, strides):
    c = int(inputs.shape[-1])
    W = tf.get_variable("W", [ksize, ksize, c, num_out], initializer=contrib.layers.xavier_initializer(), regularizer=contrib.layers.l2_regularizer(1e-5))
    b = tf.get_variable("b", [num_out], initializer=tf.constant_initializer([0.01]))
    return tf.nn.conv2d(inputs, W, [1, strides, strides, 1], "SAME") + b

def fullycon(inputs, num_out):
    c = int(inputs.shape[-1])
    W = tf.get_variable("W", [c, num_out], initializer=contrib.layers.xavier_initializer(), regularizer=contrib.layers.l2_regularizer(1e-5))
    b = tf.get_variable("b", [num_out], initializer=tf.constant_initializer([0.01]))
    return (tf.matmul(inputs, W) + b)

def global_avg_pool(inputs):
    w = int(inputs.shape[1])
    h = int(inputs.shape[2])
    return tf.nn.avg_pool(inputs, [1, w, h, 1], [1, 1, 1, 1], "VALID")

def SE_Block(inputs):
    #Squeeze-and-Excitation Networks
    #Hu J, Shen L, Sun G. Squeeze-and-Excitation Networks[J]. 2017.
    #2017年最后一届Imgenet冠军，该模块中增加了sigmoid门函数，能够加强通道之间有用的信息，抑制无用信息
    #增加该模块后欧颜柳赵四体识别率从97.2%升到了98.125%
    c = int(inputs.shape[-1])
    squeeze = tf.squeeze(global_avg_pool(inputs), [1, 2])
    with tf.variable_scope("FC1"):
        excitation = tf.nn.relu(fullycon(squeeze, int(c/16)))
    with tf.variable_scope("FC2"):
        excitation = tf.nn.sigmoid(fullycon(excitation, c))
    excitation = tf.reshape(excitation, [-1, 1, 1, c])
    scale = inputs * excitation
    return scale


def batchnorm(x, train_phase, scope_bn):
    #Batch Normalization
    #Ioffe S, Szegedy C. Batch normalization: accelerating deep network training by reducing internal covariate shift[J]. 2015:448-456.
    with tf.variable_scope(scope_bn):
        beta = tf.Variable(tf.constant(0.0, shape=[x.shape[-1]]), name='beta', trainable=True)
        gamma = tf.Variable(tf.constant(1.0, shape=[x.shape[-1]]), name='gamma', trainable=True)
        batch_mean, batch_var = tf.nn.moments(x, [0, 1, 2], name='moments')
        ema = tf.train.ExponentialMovingAverage(decay=0.5)

        def mean_var_with_update():
            ema_apply_op = ema.apply([batch_mean, batch_var])
            with tf.control_dependencies([ema_apply_op]):
                return tf.identity(batch_mean), tf.identity(batch_var)

        mean, var = tf.cond(train_phase, mean_var_with_update,
                            lambda: (ema.average(batch_mean), ema.average(batch_var)))
        normed = tf.nn.batch_normalization(x, mean, var, beta, gamma, 1e-3)
    return normed

def svm_loss(logits, labels):
    m = 1
    s_y = tf.reshape(tf.reduce_sum(logits * labels, axis=1), shape=[int(logits.shape[0]), 1])
    s_j = logits
    return tf.reduce_mean(tf.reduce_sum(tf.maximum(0., s_j - s_y + m), axis=1) - m)


class ouyanliuzhao:
    def __init__(self, batchsize=50,lr=1e-3):
        self.batchsize = batchsize
        self.lr = tf.placeholder("float")
        self.inputs = tf.placeholder("float", [batchsize, 64, 64, 1])
        self.labels = tf.placeholder("float", [batchsize, 4])
        self.training = tf.placeholder("bool")
        self.prediction = self.networks(self.inputs, self.training)
        correct_prediction = tf.equal(tf.argmax(self.prediction, 1), tf.argmax(self.labels, 1))
        self.accurancy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
        self.loss = -tf.reduce_sum(self.labels*tf.log(self.prediction+1e-12))#svm_loss(self.logits, self.labels)##tf.reduce_mean(-tf.log(tf.reduce_sum(self.logits * self.labels, axis=1)+ 1e-8))
        tf.summary.scalar("loss", self.loss)
        self.Opt = tf.train.AdamOptimizer(lr).minimize(self.loss)
        self.sess = tf.Session()
        self.merged = tf.summary.merge_all()
        self.writer = tf.summary.FileWriter("./summary/", self.sess.graph)
        self.sess.run(tf.global_variables_initializer())

    def train(self):
        lr = 1e-3
        data = sio.loadmat("./ouyanliuzhaoData.mat")
        traindata = np.reshape(data["train"], [2400, 64, 64, 1])
        trainlabel = data["train_label"]
        testdata = np.reshape(data["test"], [800, 64, 64, 1])
        testlabel = data["test_label"]
        maxacc = 0
        count = 0
        for epoch in range(1000):
            for i in range(int(2400/self.batchsize - 1)):
                self.sess.run(self.Opt, feed_dict={self.inputs: traindata[i*self.batchsize:i*self.batchsize+self.batchsize, :],
                                                   self.labels: trainlabel[i*self.batchsize:i*self.batchsize+self.batchsize, :],
                                                   self.training: True, self.lr: lr})
                result = self.sess.run(self.merged, feed_dict={
                                                    self.inputs: traindata[i * self.batchsize:i * self.batchsize + self.batchsize, :],
                                                    self.labels: trainlabel[i * self.batchsize:i * self.batchsize + self.batchsize, :],
                                                    self.training: False})
                self.writer.add_summary(result, count)
                count += 1

            trainacc = self.sess.run(self.accurancy, feed_dict={
                self.inputs: traindata[5 * self.batchsize:5 * self.batchsize + self.batchsize, :],
                self.labels: trainlabel[5 * self.batchsize:5 * self.batchsize + self.batchsize, :],
                self.training: False})
            testacc = 0
            for j in range(int(800/self.batchsize)):
                testacc += self.sess.run(self.accurancy, feed_dict={self.inputs: testdata[j*self.batchsize:j*self.batchsize+self.batchsize, :],
                                                                    self.labels: testlabel[j*self.batchsize:j*self.batchsize+self.batchsize, :], self.training: False})
            testacc = testacc/(800/self.batchsize)
            if testacc > maxacc:
                maxacc = testacc
            print("Epoch: %d Train Accurate: %g, Test Accurate: %g, MaxAcc: %g"%(epoch, trainacc, testacc, maxacc))
            k = np.array(range(2400))
            np.random.shuffle(k)
            traindata = traindata[k, :]
            trainlabel = trainlabel[k, :]


    def networks(self, inputs, training):
        with tf.variable_scope("SEnet"):
            with tf.variable_scope("conv1"):
                conv1 = conv(inputs, 32, 5, 1)
                conv1_pool = tf.nn.max_pool(conv1, [1, 3, 3, 1], [1, 2, 2, 1], "SAME")
                conv1_norm = batchnorm(conv1_pool, train_phase=training, scope_bn="bn")
                conv1_act = tf.nn.relu(conv1_norm)
                tf.summary.histogram("conv1_norm", conv1_norm)
                # se1 = SE_Block(conv1_act)
            with tf.variable_scope("conv2"):
                conv2 = conv(conv1_act, 32, 5, 1)
                conv2_pool = tf.nn.max_pool(conv2, [1, 3, 3, 1], [1, 2, 2, 1], "SAME")
                conv2_norm = batchnorm(conv2_pool, train_phase=training, scope_bn="bn")
                conv2_act = tf.nn.relu(conv2_norm)
                # se2 = SE_Block(conv2_act)
            with tf.variable_scope("conv3"):
                conv3 = conv(conv2_act, 64, 5, 1)
                conv3_pool = tf.nn.max_pool(conv3, [1, 3, 3, 1], [1, 2, 2, 1], "SAME")
                conv3_norm = batchnorm(conv3_pool, train_phase=training, scope_bn="bn")
                conv3_act = tf.nn.relu(conv3_norm)
                se3 = SE_Block(conv3_act)
            with tf.variable_scope("conv4"):
                conv4 = conv(se3, 128, 5, 1)
                conv4_pool = tf.nn.avg_pool(conv4, [1, 3, 3, 1], [1, 2, 2, 1], "SAME")
                conv4_norm = batchnorm(conv4_pool, train_phase=training, scope_bn="bn")
                conv4_act = tf.nn.relu(conv4_norm)
                se4 = SE_Block(conv4_act)
            with tf.variable_scope("fullycon"):
                #去掉全连接层用全局池化替换，对最终识别率没什么影响
                dense_inputs = tf.squeeze(global_avg_pool(se4), [1, 2])
        with tf.variable_scope("outputs"):
            logits = fullycon(dense_inputs, 4)
            prediction = tf.nn.softmax(logits)
        return prediction

if __name__ == "__main__":
    oylz = ouyanliuzhao()
    oylz.train()
