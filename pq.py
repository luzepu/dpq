from numpy.lib.shape_base import dsplit
from torch import nn
import torch
import numpy as np
from scipy.linalg import svd
import scipy.io as sio
from sklearn.cluster import KMeans
from scipy.cluster.vq import kmeans, kmeans2, vq
import random

seed = 10
torch.manual_seed(seed)
random.seed(seed)
np.random.seed(seed)

class ProductQuantization:
    def __init__(self, M, K, D):
        assert D % M == 0
        self.M = M
        self.K = K
        self.step = D // M
        self.center = np.zeros((M, K, self.step))

    def fit(self, X):
        for i in range(self.M):
            start = i * self.step
            end = (i + 1) * self.step
            subX = X[:, start:end]
            #self.center[i,:,:] = KMeans(self.K).fit(subX).cluster_centers_
            self.center[i,:,:], _ = kmeans2(subX, self.K, iter=100)
        # print(self.center.shape)
        return self.center

    def predict(self, X):
        X_ = np.zeros_like(X)
        codecode = np.zeros((X.shape[0], self.M))
        dist_g = 0
        for i in range(self.M):
            start = i * self.step
            end = (i + 1) * self.step
            subX = X[:, start:end]
            label, dist = vq(subX, self.center[i,:,:])
            codecode[:,i] = label
            # print(label, dist)
            X_[:, start:end] = self.center[i, label,:]
            dist_g += (dist ** 2).sum()
        return X_, dist_g, codecode.astype(int)

    