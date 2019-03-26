import numpy as np
import scipy.io as sio
from skimage.feature import hog

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

def get_feat(img):
    fd = hog(img, orientations=13, pixels_per_cell=[8,8], cells_per_block=[7,7], visualize=False, transform_sqrt=True,block_norm='L2-Hys')
    return fd


if __name__ == '__main__':
    data = sio.loadmat("./CV_1_801.mat")
    traindata = data["train"]
    trainlabel = data["train_label"]
    testdata = data["test"]
    testlabel = data["test_label"]
    feature_train = np.zeros([2400, 2548])
    feature_test = np.zeros([800, 2548])
    for i in range(2400):
        feature_train[i, :] = get_feat(traindata[i, :, :])
    for i in range(800):
        feature_test[i, :] = get_feat(testdata[i, :, :])

    keep_dim = 80
    FeatureVector, FinalData, mean = PCA(feature_train, keep_dim)
    traindata = np.transpose(FinalData)
    testdata = np.matmul(feature_test - mean, FeatureVector[:, :keep_dim])
    trainlabel = np.argmax(trainlabel, axis=1)
    testlabel = np.argmax(testlabel, axis=1)
    sio.savemat("hog.mat", {"traindata": traindata, "trainlabel": trainlabel, "testdata": testdata, "testlabel": testlabel})
    pass