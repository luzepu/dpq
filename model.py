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

    
class OptimizedProductQuantization(ProductQuantization):
    def __init__(self, M, K, D, maxIter=30):
        super().__init__(M, K, D)
        self.R = np.eye(D)
        self.maxIter = maxIter

    def fit(self, X):
        X_ = X @ self.R
        super().fit(X_)
        for iter in range(self.maxIter):
            Y, score, codebook = super().predict(X_)
            print(score, np.power(np.linalg.norm(X_ - Y), 2))
            [U, _, Vh] = svd(X.T @ Y) #np.linalg.svd(X.T @ Y)
            self.R = U @ Vh
            X_ = X @ self.R
            for i in range(self.M):
                start = i * self.step
                end = (i + 1) * self.step
                subX = X_[:, start:end]
                self.center[i,:,:], _ = kmeans2(subX, self.center[i,:,:], iter=10, minit='matrix')  
            
    def predict(self, X):
        return super().predict(X @ self.R)


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
        # print(dot, dot.shape)
        c_sqlen = torch.sum(center_d * center_d, -1) # M x K
        dist = c_sqlen.unsqueeze(-1) - 2 * dot + x_sqlen.unsqueeze(1)
        # print(x_sqlen.unsqueeze(1).shape, dist.shape, dot.shape)
        dist = -torch.sqrt(dist.permute(2, 0, 1)) # B x M x K
        #assign = softargmax(dist.detach(), -1, self.T)
        assign, label = softargmax(dist, -1, self.T)
        # print(dot, dist, assign)
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
        return torch.matmul(result, self.center)
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


class Duplicate(nn.Module):
    def __init__(self, m, d):
        super().__init__()
        self.m = m
        self.d = d
    def forward(self, X):
        return torch.tile(X, [self.m, 1, 1]).transpose(0,1)
    
    def merge(self, X):
        return X[:, 0, :]



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
    
class RotateSubspace(Subpace):
    def __init__(self, m, d):
        super().__init__(m, d)
        self.linear = nn.Linear(d, d, bias = False).double()
    
    def forward(self, X):
        X_ = self.linear(X)
        return super().forward(X_)


class ProjKmeans(nn.Module):
    def __init__(self, M, K, D, T):
        super().__init__()
        self.proj = Projection_share(M, D, D // M)
        #self.proj = RotateSubspace(M, D)
        self.km = MKmeansNN(M, K, self.proj.d, T)
    def forward(self, X):
        X_p = self.proj(X)
        X_r, centroid, label = self.km(X_p)
        return X_r, X_p, self.proj.merge(X_r), self.proj.merge(X_p), centroid, label


# criterion = My_loss()
class directLoss(nn.Module):
    def __init__(self):
        super(directLoss, self).__init__()   #没有需要保存的参数和状态信息

    def coreff(self, x, y):
        x_mean = torch.mean(x)
        y_mean = torch.mean(y)
        c1 = 0
        c2 = 0
        c3 = 0
        for i in range(24):
            c1 += (x - x_mean)[i] * (y - y_mean)[i]
            c2 += (x - x_mean)[i]**2
            c3 += (y - y_mean)[i]**2
        return c1/torch.sqrt(c2*c3)

    def rmse(self, i, x, y):
        sum = 0
        for i in range(x.shape[0]):
            sum += ((x[i]-y[i])**2)
        return torch.sqrt(sum/x.shape[0])
        
    def RMSE(self, x, y):
        sum = 0
        c = [1.5]*4 + [2]*7 + [3]*7 + [4]*6   # sum = 65
        for i in range(24):
            sum += c[i]*self.rmse(i, x[:, i], y[:, i])
        return sum

    def forward(self, preds, target):  # 定义前向的函数运算即可
        
#         preds = preds.cpu().detach().numpy().squeeze()
#         target = target.cpu().detach().numpy().squeeze()
#         acskill = 0
#         RMSE = self.rmse(label, preds) * 24
        
#         a = 0
#         a = [1.5]*4 + [2]*7 + [3]*7 + [4]*6 = 6+14+21+24 = 65
# #         cor = self.coreff(label, preds)
#         for i in range(24):
#             acskill += a[i] * np.log(i+1) * cor[i]
        return self.RMSE(preds, target)

def mkmeansnn(X, test, num_clusters, device, num_codebooks=8, batch_size=64, max_iter=100):
    D = X.shape[1]
    #proj = Projection(num_codebooks, D, D // num_codebooks)
    X_ = torch.tensor(X)
    test_X = torch.tensor(test)
    data_load = DataLoader(MyDataset(X), batch_size=batch_size, shuffle=True)
    loss = nn.MSELoss()
    model = ProjKmeans(num_codebooks, num_clusters, X.shape[1], 1).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.98)
    
    filename = './log/train/multipq8_128.txt'
    f = open(filename, 'w')

    test_filename = './log/test/multipq8_128.txt'
    test_f = open(test_filename, 'w')
    
    
    for i in range(max_iter):
        for batch, (X_B, y_B) in enumerate(data_load):
            X_B = X_B.to(device)
            X_r, X_p, X_r_m, X_p_m, _c, _l = model(X_B)
            # print(X_r.shape, X_p.shape, X_p_m.shape, X_B.shape)
            output = loss(X_r, X_p) + loss(X_p_m, X_B) + diversity(X_p, device)
            #output = loss(X_p, X_r) + loss(X_B, X_r_m)# + diversity(X_p)
            #output = loss(X_B, X_r.sum(1))
            #output = loss(X_B, X_p.sum(1))
            optimizer.zero_grad()
            output.backward()
            optimizer.step()
        scheduler.step()
        X_ = X_.to(device)
        X_r, X_p, X_r_m, X_p_m, centroid, label = model(X_)
        # print(centroid.shape) # 4, 20, 512
        # print(label.shape) # 64(n), 4, 1
        # output = loss(X_, X_p_m) + diversity(X_p, device)
        train_score = torch.square(X_ - X_r_m).sum().item()
        # print(i, torch.square(X_ - X_r_m).sum().item(), output.item())

        f.write(str(i)+"\t" + str(train_score)+"\n")

        with torch.no_grad():
            test_X = test_X.to(device)
            test_X_r, test_X_p, test_X_r_m, test_X_p_m, test_center, test_label = model(test_X)
            test_score = torch.square(test_X - test_X_r_m).sum().item()
            test_f.write(str(i)+"\t" + str(test_score)+"\n")

        print(i, train_score, test_score)
    return centroid, label # test_center, test_label


class MyDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index) :
        return (self.data[index].astype('float64'), 0)



def gso(X):
    # X: B x M x D
    B, M, D = X.shape
    X = X.transpose(0, 1)
    output = torch.zeros_like(X, dtype=torch.double)
    hidden = torch.zeros(B, D, D, dtype=torch.double).detach().numpy()
    for i in range(M):
        v = X[i].double()
        #print(v.dtype, hidden.dtype)
        o = v - torch.bmm(hidden, v.unsqueeze(-1)).squeeze(-1)
        output[i] = o
        #eta = o / (torch.norm(o, dim=-1, keepdim=True) + 1e-10)
        eta = o
        hidden += torch.bmm(eta.unsqueeze(-1), eta.unsqueeze(-2)).numpy()
    return output.transpose(0, 1)

## The following code tests GSO
# B, M, D = 64, 10, 200
# array = torch.randn(B, M, D)
# o = gso(array)
# o1 = o.transpose(1, 2)
# x = torch.bmm(o, o1)
# print((x - x * torch.tile(torch.eye(M), (B, 1, 1))).norm())

# data = np.array([es[0][i] for es in sio.loadmat(r'/amax/home/liuqi/DATA/data/sun397/sun397.mat')['gist'] for i in range(es[0].shape[0])])
# train_len = round(len(data)*0.8)
# train = data[:train_len]
# test = data[train_len:]

# num_clusters = 20
# ds = MyDataset(train)
# test_ds = torch.tensor(test).unsqueeze(1)
# batch_size = 64

# dim = 512
# loss = nn.MSELoss()
# km = MKmeansNN(1, num_clusters, dim, 1)
# optimizer = torch.optim.Adam(km.parameters(), lr=1e-3)
# scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)
# data_load = DataLoader(ds, batch_size=batch_size, shuffle=True)
# size = len(data_load.dataset)
# for i in range(100):
#     for batch, (X, y) in enumerate(data_load):
#         X = X.unsqueeze(1)
#         X1 = km(X)
#         #output = torch.square(X - X1).sum(-1).mean()
#         output = loss(X1, X)
#         optimizer.zero_grad()
#         output.backward()
#         optimizer.step()
#     scheduler.step()    
#     test_output = km(test_ds)
#     test_loss = torch.square(test_output - test_ds).sum()
#     print(i, test_loss.item())
