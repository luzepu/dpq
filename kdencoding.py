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
        self.rnn = nn.LSTM(input_size=10, hidden_size=20, num_layers=2) #(input_size,hidden_size,num_layers)
        
        self.weight = nn.Parameter(torch.DoubleTensor(D, D))
        nn.init.normal_(self.weight, std=0.01)
    
    def forward(self, x):
        '''
        x of shape B x M x D
        '''
        # print(x.shape)
        center_d = self.center
        x1 = x.detach().permute(1, 2, 0) # M x D x B
        x_sqlen = torch.sum(x1 * x1, 1) # M x B
        dot = torch.bmm(center_d, x1) # M x K x B
        # print(dot, dot.shape)
        c_sqlen = torch.sum(center_d * center_d, -1) # M x K
        dist = c_sqlen.unsqueeze(-1) - 2 * dot + x_sqlen.unsqueeze(1)
        # print(x_sqlen.unsqueeze(1).shape, dist.shape, dot.shape)
        dist = -torch.sqrt(dist.permute(2, 0, 1)) # B x M x K
        #assign = softargmax(dist.detach(), -1, self.T)
        assign, label = softargmax(dist, -1, self.T) # 64, 8, 32  B x M x K
        # print(assign.shape)
        return torch.bmm(assign.transpose(0,1), self.center).transpose(0, 1), center_d, label # M x B x D
        #att_logit = torch.matmul(x, self.center.T)
        #return torch.matmul(, self.center)
        #return torch.matmul(weight, self.center)
        #return self.center[att_logit.detach().argmax(-1)]


class Projection(nn.Module):
    def __init__(self, M, D_in, D_out):
        super(Projection, self).__init__()
        self.M = M
        self.drop = nn.Dropout()
        self.encoders = nn.ModuleList([nn.Linear(D_in, D_out, bias = False).double() for _ in range(M)])
        self.decoders = nn.ModuleList([nn.Linear(D_out, D_in, bias = False).double() for _ in range(M)])
        self.d = D_in

    def forward(self, x):
        output = [d(e(x)) for (e, d) in zip(self.encoders, self.decoders)]
        return torch.stack(output, dim=1)
    
    def merge(self, X):
        return X.sum(1)
        #return X.mean(1)


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


class KmeansNN(nn.Module):
    def __init__(self, K, D, T):
        super(KmeansNN, self).__init__()
        self.center = nn.Parameter(torch.DoubleTensor(K, D))
        nn.init.normal_(self.center, std=0.01)
        self.T = T
    
    def forward(self, x):
        '''
        x of shape B x D
        '''
        att_logit = -torch.sqrt(torch.sum(torch.square(x.unsqueeze(-2) - self.center), dim=-1)) / self.T
        #att_logit = torch.matmul(x, self.center.T)
        result, label = softargmax(att_logit.detach(), -1, self.T)
        return torch.matmul(result, self.center), self.center, label
        #return torch.matmul(weight, self.center)
        #return self.center[att_logit.detach().argmax(-1)]


def kmeansnn(X, num_clusters, batch_size=64, max_iter=10):
    X_ = torch.tensor(X)
    # print(X_.shape)
    data_load = DataLoader(MyDataset(X), batch_size=batch_size, shuffle=True)
    loss = nn.MSELoss()
    km = KmeansNN(num_clusters, X.shape[1], 1)
    optimizer = torch.optim.Adam(km.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)
    for i in range(max_iter):
        for batch, (X_B, y_B) in enumerate(data_load):
            X_p = km(X_B)
            output = loss(X_B, X_p)
            #output = torch.square(X_B - X_p).sum(-1).mean()
            optimizer.zero_grad()
            output.backward()
            optimizer.step()
        scheduler.step()
        
        X_p = km(X_)
        print(i, torch.square(X_ - X_p).sum().item())
        #print(i, loss(X_p, X_) * X.shape[0])
    


class ProjKmeans(nn.Module):
    def __init__(self, M, K, D, T):
        super().__init__()
        self.proj = Projection_share(M, D, D // M)
        #self.proj = RotateSubspace(M, D)
        self.km = KmeansNN(K, self.proj.d, T)
    def forward(self, X):
        X_p = self.proj(X)
        X_r, centroid, label = self.km(X_p)
        return X_r, X_p, self.proj.merge(X_r), self.proj.merge(X_p), centroid, label

class Projection_DNN(nn.Module):
    def __init__(self, M, D_in, D_out):
        super().__init__()
        self.M = M
        self.weight = nn.Parameter(torch.DoubleTensor(M, D_in, D_out))
        nn.init.normal_(self.weight, std=0.01)
        #nn.init.xavier_normal_(self.weight)
        self.d = D_in
    def forward(self, X):
        # B x D(K), return B x M x D
        # c = onehot(x), d = tempering softmax(c), 
        # y1 = torch.bmm(self.weight, c), y2 = rnn(self.weight, c), return d, y

        a = torch.tensordot(self.weight, X,  dims=[[1],[1]]) # M x D1 x B
        b = torch.bmm(self.weight, a) # M x D x B
        return b.permute(2, 0, 1)

class Projection_RNN(nn.Module):
    def __init__(self, M, D_in, D_out):
        super().__init__()
        self.M = M
        self.weight = nn.Parameter(torch.DoubleTensor(M, D_in, D_out))
        nn.init.normal_(self.weight, std=0.01)
        #nn.init.xavier_normal_(self.weight)
        self.d = D_in
    def forward(self, X):
        # B x D(K), return B x M x D
        # c = onehot(x), d = tempering softmax(c), 
        # y1 = torch.bmm(self.weight, c), y2 = rnn(self.weight, c), return d, y

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
    

class KD_Encoding(nn.Module):
    def __init__(self, M, K, D, T):
        super().__init__()
        self.dup = Duplicate(M)
        self.proj = Projection_RNN(M, D, D // M)
        self.kd = MKmeansNN(M, K, D//M, T)
        self.M = M
        self.weight = nn.Parameter(torch.DoubleTensor(D, D))
        nn.init.normal_(self.weight, std=0.01)
        self.sub = Subpace(M, D)
        self.M = M

    def forward(self, X, m):
        U = X
        # X = self.dup(X)
        X = self.sub(X) # B x M x D//M
        X_p, center, codes = self.kd(X) # B x M x D//M ;  M x K x D//M ; B x M x 1
        # print(X_p.shape, center.shape, codes.shape)
        # d, X_r = self.proj(X_p)
        X_p = X_p.permute(1,0,2)
        for i in range(self.M):
            if i == 0:
                X_p_m = X_p[i]
            else:
                X_p_m = torch.cat((X_p_m, X_p[i]), 1)
        # X_p_m = self.proj.merge(X_p)
        # print(X_p_m.shape)
        # X_p_m = torch.matmul(X_p_m, self.weight)
        # print(center.shape)
        X_p = X_p.permute(1,0,2)
        return X_p, X_p_m, center, codes, self.weight
        # print(X_p_m.shape)
        # Y = torch.mul(X_p_m, m) + torch.mul(U, 1-m)

        # return X_p, Y, center, codes, self.weight, U, X_p_m 


def kdencodingnn(X, test, num_clusters, device, num_codebooks, batch_size=32, max_iter=100):
    D = X.shape[1]
    #proj = Projection(num_codebooks, D, D // num_codebooks)
    X_ = torch.tensor(X)
    test_X = torch.tensor(test)
    data_load = DataLoader(MyDataset(X), batch_size=batch_size, shuffle=True)
    loss = nn.MSELoss()
    entroy_loss = nn.CrossEntropyLoss()
    model = KD_Encoding(num_codebooks, num_clusters, X.shape[1], 1).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.98)
    
    filename = './log/kdencoding-codebooks=8.txt'
    test_filename = './log/kdencoding_test_loss-codebooks=8.txt'
    f = open(filename, 'w')
    test_f = open(test_filename, 'w')
    
    
    for i in range(max_iter):
        for batch, (X_B, y_B) in enumerate(data_load):
            X_B = X_B.to(device)
            _, X_r, center, codes, H= model(X_B, 0.7)
            # print(X_r.shape, center.shape, codes.shape, X_B.shape)
            output = loss(X_r, X_B) #+ loss(U, FC) + entroy_loss(d)
            optimizer.zero_grad()
            output.backward()
            optimizer.step()
        scheduler.step()

        # with torch.no_grad():
        X_ = X_.to(device)
        _, X_r, center, codes, H = model(X_, 1)
        # print(centroid.shape) # 4, 20, 512
        # print(label.shape) # 64(n), 4, 1
        output = loss(X_r, X_) # + entroy_loss(d)
        train_score = torch.square(X_ - X_r).sum().item()
        # print(i, torch.square(X_ - X_r_m).sum().item(), output.item())
        f.write(str(i)+"\t" + str(train_score)+"\n")

        test_X = test_X.to(device)
        _, test_X_r, test_center, test_codes, test_H = model(test_X, 1)
        test_score = torch.square(test_X - test_X_r).sum().item()
        test_f.write(str(i)+"\t" + str(test_score)+"\n")

        print(i, train_score, test_score)

    return center, codes, H

class MyDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index) :
        return (self.data[index].astype('float64'), 0)
