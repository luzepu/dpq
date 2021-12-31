from numpy.lib.shape_base import dsplit
from torch import nn
import torch
import numpy as np
import cupy as cp
from scipy.linalg import svd
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
        # self.center = nn.Parameter(torch.DoubleTensor(M, K, self.step))
        # nn.init.normal_(self.center, std=0.01)

    def fit(self, X):
        for i in range(self.M):
            start = i * self.step
            end = (i + 1) * self.step
            subX = X[:, start:end]
            #self.center[i,:,:] = KMeans(self.K).fit(subX).cluster_centers_
            # print(subX, self.K)
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
            # print(dist)
            codecode[:,i] = label
            # print(label, dist)
            X_[:, start:end] = self.center[i, label,:]
            dist_g += (dist ** 2).sum()
        return X_, dist_g, codecode.astype(int)

    
    
class OptimizedProductQuantization(ProductQuantization):
    def __init__(self, M, K, D, maxIter=64):
        super().__init__(M, K, D)
        self.R = np.eye(D)
        self.maxIter = maxIter
        # self.center = np.ones((M, K, self.step))
        # self.center = nn.Parameter(torch.DoubleTensor(M, K, self.step))
        # nn.init.normal_(self.center, std=0.01)

    def fit(self, X, test):
        X_ = X @ self.R
        test_ = test @ self.R
        super().fit(X_)
        for iter in range(self.maxIter):
            Y, score, codebook = super().predict(X_)
            test_Y, test_score, test_codebook = super().predict(test_)
            # print(score, np.power(np.linalg.norm(X_ - Y), 2))
            print(iter, score, test_score)
            [U, _, Vh] = np.linalg.svd(X.T @ Y) # svd(X.T @ Y)  XT*center = u*_*v;  CENTERT*X = V*_T*UT
            self.R = U @ Vh
            X_ = X @ self.R
            test_ = test @ self.R
            for i in range(self.M):
                start = i * self.step
                end = (i + 1) * self.step
                subX = X_[:, start:end]
                # print(subX.shape, self.center[i,:,:].shape)
                self.center[i,:,:], label = kmeans2(subX, self.center[i,:,:], iter=30, minit='matrix')
        return self.R, self.center

    def predict(self, X):
        return super().predict(X @ self.R)


# class ProjKmeans(nn.Module):
#     def __init__(self, M, K, D, T):
#         super().__init__()
#         self.proj = ProductQuantization(M, K, D)
#     def forward(self, X):
#         X_p = self.proj(X)
#         return X_p
