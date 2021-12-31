from numpy.lib.shape_base import dsplit
from torch import nn
import torch
import numpy as np
from scipy.linalg import svd
import scipy.io as sio
import math
from torch._C import dtype
from torch.utils.data import DataLoader, Dataset
from sklearn.cluster import KMeans
from scipy.cluster.vq import kmeans, kmeans2, vq
import random

seed = 10
torch.manual_seed(seed)
random.seed(seed)
np.random.seed(seed)

def diversity(X, device):
    m = X.shape
    XX = torch.bmm(X, X.transpose(-2, -1))
    xnorm = torch.norm(X, dim=-1)
    cossim = XX / torch.bmm(xnorm.unsqueeze(-1), xnorm.unsqueeze(1))
    
    cossim_off =  cossim - torch.tile(torch.eye(m[1]).to(device), (X.shape[0], 1, 1))
    # print(cossim, torch.eye(m[1]), X.shape[0])
    return cossim_off.square().sum() / m[0] / (m[1]**2 - m[1])

def softargmax(logits, dim, T):
    y_soft = torch.softmax(logits/T, dim)
    index = y_soft.max(dim, keepdim=True)[1]
    label = index
    y_hard = torch.zeros_like(logits, memory_format=torch.legacy_contiguous_format).scatter_(dim, index, 1.0)
    # print(y_hard, y_hard.shape)
    result = y_hard - y_soft.detach() + y_soft
    # print(result, result.shape, label.shape)
    return result, label


class MKmeansNN(nn.Module):
    def __init__(self, M, K, D, T):
        super(MKmeansNN, self).__init__()
        self.center = nn.Parameter(torch.DoubleTensor(M, K, D))
        nn.init.normal_(self.center, std=0.01)
        self.T = T
    
    def forward(self, x):
        '''
        x of shape B x M x D
        '''
        # print(x.shape)
        center_d = self.center
        x1 = x.detach().permute(1, 2, 0) # M x D x B
        x_sqlen = torch.sum(x1 * x1, 1) # M x B
        dot = torch.bmm(center_d, x1) # M x K x B
        c_sqlen = torch.sum(center_d * center_d, -1) # M x K
        dist = c_sqlen.unsqueeze(-1) - 2 * dot + x_sqlen.unsqueeze(1)
        dist = -torch.sqrt(dist.permute(2, 0, 1)) # B x M x K
        
        assign, label = softargmax(dist, -1, self.T) # 64, 8, 32  B x M x K
        return torch.bmm(assign.transpose(0,1), self.center).transpose(0, 1), center_d, label # M x B x D
        


class Projection_share(nn.Module):
    def __init__(self, M, D_in, D_out):
        super().__init__()
        self.M = M
        self.weight = nn.Parameter(torch.DoubleTensor(M, D_in, D_out))
        nn.init.normal_(self.weight, std=0.01)
        #nn.init.xavier_normal_(self.weight)
        self.d = D_in
    def forward(self, X):
        # B x D, M x D x D1, return B x M x D
        a = torch.tensordot(self.weight, X,  dims=[[1],[1]]) # M x D1 x B
        b = torch.bmm(self.weight, a) # M x D x B
        return b.permute(2, 0, 1)
    
    def merge(self, X):
        return X.sum(1)



class Duplicate(nn.Module):
    def __init__(self, m):
        super().__init__()
        self.m = m
    def forward(self, X):
        X = X.unsqueeze(1)
        tmp = X
        for i in range(1, self.m):
            tmp = torch.cat((X, tmp), 1)
        return tmp

class Subpace(nn.Module):
    def __init__(self, m, d):
        super().__init__()
        assert d % m == 0
        self.m = m
        self.d = d // m
    
    def forward(self, X):
        B = X.size(0)
        return X.view(B, self.m, -1)
    
    def merge(self, X):
        B = X.size(0)
        return X.reshape(B, -1)
    

class DiffPQ(nn.Module):
    def __init__(self, M, K, D, T, metric):
        super().__init__()
        self.sub = Subpace(M, D)
        self.kd = MKmeansNN(M, K, D//M, T)
        self.M = M
        self.loss = nn.MSELoss()

    def forward(self, X):
        X_p = self.sub(X) # B x M x D//M
        X_r, centroid, label = self.kd(X_p) # B x M x D//M ;  M x K x D//M ; B x M x 1
        X_r = X_r.permute(1,0,2)
        for i in range(self.M):
            if i == 0:
                X_r_m = X_r[i]
            else:
                X_r_m = torch.cat((X_r_m, X_r[i]), 1)
        X_r = X_r.permute(1,0,2)
        output = self.loss(X_r, X_p) + self.loss(X_r_m, X)
        return X_r, X_p, X_r_m, X, centroid, label, output
        

def DiffPQuantization(X, test, num_clusters, device, num_codebooks, batch_size=64, max_iter=100):
    D = X.shape[1]
    #proj = Projection(num_codebooks, D, D // num_codebooks)
    X_ = torch.tensor(X)
    test_X = torch.tensor(test)
    data_load = DataLoader(MyDataset(X), batch_size=batch_size, shuffle=True)
    loss = nn.MSELoss()
    entroy_loss = nn.CrossEntropyLoss()
    model = DiffPQ(num_codebooks, num_clusters, X.shape[1], 1).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.98)
    
    for i in range(max_iter):
        for batch, (X_B, y_B) in enumerate(data_load):
            X_B = X_B.to(device)
            X_r, X_p, X_r_m, X_p_m, centroid, label, output = model(X_B)
            # print(X_r.shape, center.shape, codes.shape, X_B.shape)
            # output = loss(X_r, X_B) #+ loss(U, FC) + entroy_loss(d)
            # output += loss(X_B, X_r_m)
            optimizer.zero_grad()
            output.backward()
            optimizer.step()
        scheduler.step()

        # with torch.no_grad():
        X_ = X_.to(device)
        X_r, X_p, X_r_m, X_p_m, centroid, label, output = model(X_)
        # print(centroid.shape) # 4, 20, 512
        # print(label.shape) # 64(n), 4, 1
        # output = loss(X_r, X_) # + entroy_loss(d)
        train_score = torch.square(X_ - X_r_m).sum().item()
        # print(i, torch.square(X_ - X_r_m).sum().item(), output.item())

        test_X = test_X.to(device)
        test_X_r, test_X_p, test_X_r_m, test_X_p_m, test_centroid, test_label, test_output = model(test_X)
        test_score = torch.square(test_X - test_X_r_m).sum().item()

        print(i, train_score, test_score)

    return centroid, test_label

class MyDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index) :
        return (self.data[index].astype('float64'), 0)
