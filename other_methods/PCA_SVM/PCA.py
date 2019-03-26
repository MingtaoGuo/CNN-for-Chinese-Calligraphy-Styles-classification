import numpy as np
from PIL import Image
import scipy.io as sio
import matplotlib.pyplot as plt


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

def reconstruction(data, FeatureVector, mean, keep_dim):
    #data为一个行向量样本
    w = np.matmul(data-mean, FeatureVector[:, :keep_dim])
    new_data = mean + np.transpose(np.sum(w * FeatureVector[:, :keep_dim], axis=1, keepdims=True))
    return new_data

if __name__ == "__main__":
    data = sio.loadmat("./CV_1_801.mat")
    traindata = np.reshape(data["train"], [-1, 64*64])
    trainlabel = np.argmax(data["train_label"], axis=1)
    testdata = np.reshape(data["test"], [-1, 64*64])
    testlabel = np.argmax(data["test_label"], axis=1)
    keep_dim = 140
    FeatureVector, FinalData, mean = PCA(traindata, keep_dim)
    traindata = np.transpose(FinalData)
    testdata = np.matmul(testdata - mean, FeatureVector[:, :keep_dim])
    sio.savemat("pca.mat", {"traindata": traindata, "trainlabel": trainlabel, "testdata": testdata, "testlabel": testlabel})
    pass
