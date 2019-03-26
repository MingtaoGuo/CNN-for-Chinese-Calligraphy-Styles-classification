from networks import network, resnet18, alexnet, vgg16, googlenet
from utils import random_read_batch
import tensorflow as tf
import scipy.io as sio
import numpy as np


BATCH_SIZE = 50
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-3
EPSILON = 1e-10

def train():
    lr = tf.placeholder("float")
    inputs = tf.placeholder("float", [None, 64, 64, 1])
    labels = tf.placeholder("float", [None, 4])
    is_training = tf.placeholder("bool")
    prediction = network(inputs, is_training)
    correct_prediction = tf.equal(tf.argmax(prediction, 1), tf.argmax(labels, 1))
    accurancy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
    loss = -tf.reduce_sum(labels * tf.log(prediction + EPSILON)) + tf.add_n(
        [tf.nn.l2_loss(var) for var in tf.trainable_variables()]) * WEIGHT_DECAY
    Opt = tf.train.AdamOptimizer(lr).minimize(loss)
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    data = sio.loadmat("./CV_1_801.mat")
    traindata = np.reshape(data["train"], [2400, 64, 64, 1]) / 127.5 - 1.0
    trainlabel = data["train_label"]
    testdata = np.reshape(data["test"], [800, 64, 64, 1]) / 127.5 - 1.0
    testlabel = data["test_label"]
    max_test_acc = 0
    loss_list = []
    acc_list = []
    for i in range(11000):
        batch_data, label_data = random_read_batch(traindata, trainlabel, BATCH_SIZE)
        sess.run(Opt, feed_dict={inputs: batch_data, labels: label_data, is_training: True, lr: LEARNING_RATE})
        if i % 20 == 0:
            [LOSS, TRAIN_ACCURACY] = sess.run([loss, accurancy], feed_dict={inputs: batch_data, labels: label_data, is_training: False, lr: LEARNING_RATE})
            loss_list.append(LOSS)
            TEST_ACCURACY = 0
            for j in range(16):
                TEST_ACCURACY += sess.run(accurancy, feed_dict={inputs: testdata[j*50:j*50+50], labels: testlabel[j*50:j*50+50], is_training: False, lr: LEARNING_RATE})
            TEST_ACCURACY /= 16
            acc_list.append(TEST_ACCURACY)
            if TEST_ACCURACY > max_test_acc:
                max_test_acc = TEST_ACCURACY
            print("Step: %d, loss: %4g, training accuracy: %4g, testing accuracy: %4g, max testing accuracy: %4g"%(i, LOSS, TRAIN_ACCURACY, TEST_ACCURACY, max_test_acc))
        if i % 1000 == 0:
            np.savetxt("loss_list.txt", loss_list)
            np.savetxt("acc_list.txt", acc_list)


if __name__ == "__main__":
    train()