import re
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


def softargmax(logits, dim, T):
    y_soft = torch.softmax(logits/T, dim)
    
    index = y_soft.max(dim, keepdim=True)[1]
    label = index
    y_hard = torch.zeros_like(logits, memory_format=torch.legacy_contiguous_format).scatter_(dim, index, 1.0)
    # print(y_hard, y_hard.shape)
    # print(y_hard.grad)
    result = y_hard - y_soft.detach() + y_soft
    # print(result, result.shape, label.shape)
    # return result, label
    return result, label, y_soft



class KmeansNN(nn.Module):
    def __init__(self, query, M, K, D, T, metric, e):
        super(KmeansNN, self).__init__()
        self.center = nn.Parameter(torch.DoubleTensor(M, K, D))
        # self.center = nn.Parameter(torch.DoubleTensor(K, D))
        nn.init.normal_(self.center, std=0.01)
        self.T = T
        self.M = M
        self.weight = nn.Parameter(torch.DoubleTensor(1, M))
        # nn.init.normal_(self.weight, mean=0, std=1)
        nn.init.constant_(self.weight, val=1)
        # self.loss = nn.MSELoss()
        self.metric = metric
        self.e = e
        self.query = query.transpose(0, 1)
    
    def forward(self, x):
        '''
        x of shape B x D
        '''
        B = x.shape[0]
        if self.metric=='l2_distance':
            att_logit = -torch.sqrt(torch.sum(torch.square(x.unsqueeze(-2) - self.center[0]), dim=-1)) / self.T
        elif self.metric=='dot_product':
            gt = x @ self.query
            cq = self.center[0] @ self.query
            att_logit = (gt.unsqueeze(-2).repeat(1, self.K, 1) - cq.unsqueeze(-3).repeat(B, 1, 1))
            att_logit = -(att_logit * att_logit).sum(-1) / self.T

            # att_logit = self.logit(x, self.center[0])
        # att_logit = torch.cosine_similarity(x.unsqueeze(-2), self.center[0], dim = -1)
        # att_logit = torch.matmul(x, self.center[0].T)

        # NEQ
        # att_logit = self.logit(x, self.center[0])

        result, label, soft = softargmax(att_logit, -1, self.T)
        # codebooks = self.center[0].detach().cpu().numpy()
        # codes = label.detach().cpu().numpy()
        codebooks = self.center[0]
        codes = label
        RX = x
        X_p = torch.matmul(result, self.center[0])
        X_p_m = X_p
        output = self.loss(torch.matmul(result, self.center[0]), RX).view(1,1)
        soutput = self.loss(torch.matmul(soft, self.center[0]), RX).view(1,1)
        match = self.loss(result, soft).view(1,1)

        X_p_matirx = RX.unsqueeze(1)
        X_r_matirx = X_p.unsqueeze(1)

        # output = torch.DoubleTensor(self.M, 1)
        for j in range(1, self.M):
            RX = x - X_p_m
            if self.metric=='l2_distance':
                att_logit = -torch.sqrt(torch.sum(torch.square(RX.unsqueeze(-2) - self.center[j]), dim=-1)) / self.T
            elif self.metric=='dot_product':
                gt = RX @ self.query
                cq = self.center[j] @ self.query
                att_logit = (gt.unsqueeze(-2).repeat(1, self.K, 1) - cq.unsqueeze(-3).repeat(B, 1, 1))
                att_logit = -(att_logit * att_logit).sum(-1) / self.T

                # att_logit = self.logit(RX, self.center[j])
            # att_logit = -torch.sqrt(torch.sum(torch.square(RX.unsqueeze(-2) - self.center[j]), dim=-1)) / self.T
            # att_logit = torch.cosine_similarity(RX.unsqueeze(-2), self.center[j], dim = -1)
            # att_logit = torch.matmul(RX, self.center[j].T)
            # att_logit = self.logit(RX, self.center[j])
            result, label, soft = softargmax(att_logit, -1, self.T)
            X_p = torch.matmul(result, self.center[j])
            centroid = self.center[j]
            codebooks = torch.cat((codebooks, centroid), 0)
            codes = torch.cat((codes, label), 1)
            # temp = centroid.detach().cpu().numpy()
            # codebooks = np.r_[codebooks, temp]
            X_p_matirx = torch.cat((X_p_matirx, RX.unsqueeze(1)), 1)
            X_r_matirx = torch.cat((X_r_matirx, X_p.unsqueeze(1)), 1)
            # print(j, codes.shape, code.shape)
            # codes = np.c_[codes, label]
            X_p_m += X_p
            output = torch.cat((output, self.loss(torch.matmul(result, self.center[j]), RX).view(1,1)), 0)
            soutput = torch.cat((soutput, self.loss(torch.matmul(soft, self.center[j]), RX).view(1,1)), 0)
            match = torch.cat((match, self.loss(result, soft).view(1,1)), 0)
                
        # return torch.matmul(result, self.center), self.center, label
        # distortion = self.loss(X_p_m, x)

        lquanH = torch.matmul(self.weight, output)
        lquan = torch.matmul(self.weight, soutput)
        lmatch = torch.matmul(self.weight, match)

        output = lquanH + 0.1 * lmatch + lquan
        # print(output)
        return X_r_matirx, X_p_matirx, X_p_m, X_p_m, codebooks, codes, output
        #return torch.matmul(weight, self.center)
        #return self.center[att_logit.detach().argmax(-1)]

    def logit(self, x, x_bar):
        x_norm = torch.norm(x, dim=-1).unsqueeze(-1)
        center_norm = torch.norm(x_bar, dim=-1).unsqueeze(0)
        norm_err = torch.abs(x_norm-center_norm)/x_norm # B K
        direct_err =  1 - torch.matmul(x, x_bar.T) / (x_norm * center_norm) # B K
        att_logit = -(self.e * norm_err + direct_err)
        # print(norm_err, direct_err)
        # print(att_logit)
        return att_logit

    def loss(self, x, x_bar):
        if self.metric=='l2_distance':
            score = nn.MSELoss()
            att_logit = score(x, x_bar)
        elif self.metric=='dot_product':

            gt = x @ self.query
            pt = x_bar @  self.query
            loss = nn.MSELoss()
            att_logit = loss(gt, pt)

            # x_norm = torch.norm(x, dim=-1).unsqueeze(-1) # B 1
            # x_bar_norm = torch.norm(x_bar, dim=-1).unsqueeze(-1) # B 1
            # norm_err = torch.abs(x_norm-x_bar_norm)/x_norm # B 1
            # # inner_product = torch.tensordot(x_norm, x_bar_norm, dims=[[1], [1]])
            # direct_err =  1 - torch.matmul(x.unsqueeze(-1).permute(0, 2, 1), x_bar.unsqueeze(-1)) / (x_norm * x_bar_norm) # B 1
            # att_logit = (self.e * norm_err + direct_err)
        return att_logit.sum()




def DeepProgressiveQuantization(X, test, num_clusters, B, device, batch_size=64, max_iter=100):
    # X_ = torch.tensor(X)
    data_load = DataLoader(MyDataset(X), batch_size=batch_size, shuffle=False)
    loss = nn.MSELoss()
    km = KmeansNN(B, num_clusters, X.shape[1], 1).to(device)
    # km = torch.nn.DataParallel(km).cuda()
    # for n in km.state_dict():
    #     print(n)
    optimizer = torch.optim.Adam([
         {'params': km.center, 'lr': 1e-3},  # center
         {'params': km.weight, 'lr': 1e-5}]) # weight
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)
    for i in range(max_iter):
        train_loss = 0
        for batch, (X_B, y_B) in enumerate(data_load):
            X_B = X_B.to(device)
            X_p, X_r, X_p_m, centroid, code, lquan, lquanH, lmatch = km(X_B)
            # print(X_p.shape, X_r.shape, X_p_m.shape)
            if train_loss==0:
                train_code = code
            else:
                train_code = torch.cat((train_code, code), 0)
            # centroid.shape  # shape = (256,128)
            # print(centroid.shape, code.shape)
            # X_p = km(X_B)
            # print(X_B.shape, X_p.shape)
            # output = loss(X_B, X_p)
            output = lquanH + 0.1 * lmatch + lquan

            # gama = (torch.abs(torch.norm(X_r, dim=2) - torch.norm(X_p, dim=2))/torch.norm(X_p, dim=2))
            # inner_product = torch.matmul(X_r.unsqueeze(-2), X_p.unsqueeze(-1))
            # yita = (1 - inner_product.squeeze(-1).squeeze(-1)/(torch.norm(X_r, dim=2)*torch.norm(X_p, dim=2)))
            # output = (gama + 0.6 * yita).sum()

            train_loss += torch.square(X_B - X_p_m).sum().item()
            
            # x_norm = torch.norm(X_B, dim=-1)
            # x_bar_norm = torch.norm(X_p_m, dim=-1)
            # gama = (torch.abs(x_norm - x_bar_norm)/x_norm)
            # inner_product = X_B @ X_p_m.T
            # yita = (1 - inner_product/(x_norm*x_bar_norm))
            # output = (gama + 0.6 * yita).sum()

            # train_loss += output.item()
            optimizer.zero_grad()
            output.backward()
            optimizer.step()
        scheduler.step()

        # data_load = DataLoader(MyDataset(X), batch_size=batch_size, shuffle=False)
        # train_loss = 0
        
        # with torch.no_grad():
            # for batch, (X_B, y_B) in enumerate(data_load):
            #     X_B = X_B.to(device)
            #     X_p, _centroid, _code = km(X_B)
            #     if train_loss==0:
            #         # train_centroid = _centroid
            #         train_code = _code
            #     else:
            #         # train_centroid = torch.concat((train_centroid, _centroid), 1)
            #         train_code = torch.cat((train_code, _code), 0)
            #         # print(train_code.shape, _code.shape)
            #     train_loss += torch.square(X_B - X_p).sum().item()
        # print(i, train_loss)

        with torch.no_grad():
            test_loss = 0
            # test = torch.tensor(test)
            # X_ = test.to(device)
            test_data_load = DataLoader(MyDataset(test), batch_size=batch_size, shuffle=False)
            for batch, (X_, y_) in enumerate(test_data_load):
                X_ = X_.to(device)
                X_p, X_r, X_p_m, _centroid, _code, lquan, lquanH, lmatch = km(X_)
                if test_loss==0:
                    test_code = _code
                else:
                    test_code = torch.cat((test_code, _code), 0)

                # x_norm = torch.norm(X_, dim=-1)
                # x_bar_norm = torch.norm(X_p, dim=-1)
                # gama = (torch.abs(torch.norm(X_r, dim=2) - torch.norm(X_p, dim=2))/torch.norm(X_p, dim=2))
                # inner_product = torch.matmul(X_r.unsqueeze(-2), X_p.unsqueeze(-1))
                # # print(inner_product.shape)
                # yita = (1 - inner_product.squeeze(-1).squeeze(-1)/(torch.norm(X_r, dim=2)*torch.norm(X_p, dim=2)))
                # # print(gama, yita)
                # test_loss += (gama + 0.6 * yita).sum().item()
                # gama = (torch.abs(x_norm - x_bar_norm)/x_norm).sum().item()
                # inner_product = X_ @ X_p.T
                # yita = (1 - inner_product/(x_norm*x_bar_norm)).sum().item()
                # test_loss += gama + 0.6 * yita

                test_loss += torch.square(X_ - X_p_m).sum().item()
        
        print(i, train_loss, test_loss)
            # print(i, loss(X_p, X_) * X.shape[0])
            # print(centroid.shape, codes.shape)
    
    return centroid, test_code#train_code


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
    


def lightrec(X, test, num_clusters, device, num_codebooks=8, batch_size=64, max_iter=100):
    D = X.shape[1]
    #proj = Projection(num_codebooks, D, D // num_codebooks)
    X_ = torch.tensor(X)
    test_X = torch.tensor(test)
    data_load = DataLoader(MyDataset(X), batch_size=batch_size, shuffle=True)
    loss = nn.MSELoss()
    model = ProjKmeans(num_codebooks, num_clusters, X.shape[1], 1).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.98)
    
    filename = f'./log/lightrec=8.txt'
    test_filename = f'./log/lightrec-codebooks=8.txt'
    f = open(filename, 'w')
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
        output = loss(X_, X_p_m) + diversity(X_p, device)
        train_score = torch.square(X_ - X_r_m).sum().item()
        # print(i, torch.square(X_ - X_r_m).sum().item(), output.item())

        f.write(str(i)+"\t" + str(train_score)+"\n")

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
