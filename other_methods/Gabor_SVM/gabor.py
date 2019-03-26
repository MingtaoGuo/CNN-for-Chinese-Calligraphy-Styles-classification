import numpy as np
from scipy.signal import convolve2d
import tensorflow as tf
import scipy.io as sio
import matplotlib.pyplot as plt

PI = 3.14

def PCA(data, keep_dim):
    """
    :param data: 每一行为一个样本.
    :param keep_dim: 保留得维数
    :return: Eigenvectors, FinalData, mean
    假设data为m行n列(mxn)， 每一行表示一个样本
    """
    mean = np.mean(data, axis=0, keepdims=True)#1xn
    DataAdjust = data - mean#mxn
    cov = np.cov(np.transpose(DataAdjust))#nxn
    eigenvalues, eigenvector = np.linalg.eigh(cov)
    FeatureVector = np.flip(eigenvector, axis=1)#每一列为一个特征向量nxn
    RowFeatureVector = np.transpose(FeatureVector)#每一行为一个特征向量nxn
    ColDataAdjust = np.transpose(DataAdjust)#每一列为一个样本nxm
    FinalData = np.matmul(RowFeatureVector[:keep_dim, :], ColDataAdjust)#keep_dim x m
    return FeatureVector, FinalData, mean

def getGabourKernel(r, c, sigma, gamma, theta, f, u):
    sigma2 = sigma * sigma
    k = 1 / (2 * PI * gamma * sigma2)
    GK = np.zeros([r, c])
    for m in range(-r//2+1, r//2):
        for n in range(-c//2+1, c//2):
            x1 = m * np.cos(theta * u) + n * np.sin(theta * u)
            y1 = -m * np.sin(theta * u) + n * np.cos(theta * u)
            GK[m + r//2, n + c//2] = k * np.exp(-0.5 * (1 / sigma2) * ((x1 / gamma)**2 + (y1)**2)) * np.cos(2 * PI * f * x1)
    return GK

def GaborKernel(r, c, theta, a, fmax, gamma):
    kernel = np.zeros([r, c, 40])
    k = 0
    for v in range(8):
        f = a ** (-v) * fmax
        sigma = 0.56 / f
        for u in range(5):
            GK = getGabourKernel(r, c, sigma, gamma, theta, f, u)
            kernel[:, :, k] = GK
            k += 1
    return kernel

def feature_extraction(traindata, testdata, GK, down_sample=64):
    nums_k = GK.shape[-1]
    h, w = traindata.shape[1], traindata.shape[2]
    kernel = np.reshape(GK, [64, 64, 1, 40])

    inputs = tf.placeholder(tf.float32, [None, 64, 64, 1])
    conv = tf.nn.conv2d(inputs, kernel, [1, 1, 1, 1], "SAME")
    sess = tf.Session()

    train_gabor_data = np.zeros([2400, 64, 64, 40])
    test_gabor_data = np.zeros([800, 64, 64, 40])
    for i in range(12):
        train_gabor_data[i*200:i*200+200] = sess.run(conv, feed_dict={inputs: traindata[i*200:i*200+200, :, :, np.newaxis]})
    for i in range(4):
        test_gabor_data[i * 200:i * 200 + 200] = sess.run(conv, feed_dict={inputs: testdata[i * 200:i * 200 + 200, :, :, np.newaxis]})
    train_feature = np.reshape(train_gabor_data, [2400, -1])
    test_feature = np.reshape(test_gabor_data, [800, -1])
    feature_len = h*w*nums_k//down_sample
    down_sampled_train_feature = np.zeros([2400, feature_len])
    down_sampled_test_feature = np.zeros([800, feature_len])
    for i in range(feature_len):
        down_sampled_train_feature[:, i] = train_feature[:, down_sample * i]
        down_sampled_test_feature[:, i] = test_feature[:, down_sample * i]
    return down_sampled_train_feature, down_sampled_test_feature

if __name__ == "__main__":
    r = 64
    c = 64
    theta = 3.14 / 5
    a = np.sqrt(2)
    fmax = 0.22
    gamma = 0.5
    data = sio.loadmat("./CV_1_801.mat")
    traindata = data["train"]
    trainlabel = data["train_label"]
    testdata = data["test"]
    testlabel = data["test_label"]
    kernel = GaborKernel(r, c, theta, a, fmax, gamma)

    train_gabor_data, test_gabor_data = feature_extraction(traindata, testdata, kernel, down_sample=256)

    keep_dim = 80
    FeatureVector, FinalData, mean = PCA(train_gabor_data, keep_dim)
    traindata = np.transpose(FinalData)
    testdata = np.matmul(test_gabor_data - mean, FeatureVector[:, :keep_dim])
    trainlabel = np.argmax(trainlabel, axis=1)
    testlabel = np.argmax(testlabel, axis=1)
    sio.savemat("gabor.mat", {"traindata": traindata, "trainlabel": trainlabel, "testdata": testdata, "testlabel": testlabel})
    pass