from math import gamma
import numpy as np
import scipy as sp
from scipy import optimize
import scipy.io as sio
from sklearn.cluster import KMeans
import random
import torch
torch.set_num_threads(4)
# from evaluationRecall import Recall_PQ
from evaluationRecall import SearchNeighbors_AQ, SearchNeighbors_PQ, recall_atN
from torch.optim import optimizer # Recall_AQ
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import StepLR
import tqdm
import copy

from sun397 import Sun397
from model import mkmeansnn
from lightrec import LightRec
from kdencoding import kdencodingnn
from opq import ProductQuantization, OptimizedProductQuantization
from DeepPQ import KmeansNN
from SPQ3 import SoftPQuantization
from DiffPQ import DiffPQ
from diverse_direct import directPQ
from diffdiversedirect import ProjKmeans, directloss
# from tripleDPQ import D3PQ
from skewkmeans import sD3PQ
from dotmultipq import dotmultipq
# from multipq import ProjKmeans
from scann import ScaNN

seed = 10
random.seed(seed)
np.random.seed(seed)


def get_dataset(name, path):
    if name == 'sun397':
        data = np.array([es[0][i] for es in sio.loadmat(path)['gist'] for i in range(es[0].shape[0])])  # 108754

        rng = np.random.RandomState(seed)
        ind_shuf = list(range(len(data)))
        rng.shuffle(ind_shuf)
        data = data[ind_shuf]

        query_len = round(len(data)*0.02)
        query = data[:query_len]

        database = data[query_len:]
        train_len = round(len(database)*0.1)
        train = database[:train_len]
        # print(data[0])
        return query, database[train_len:], train, Sun397(train), Sun397(database[train_len:])
    elif name=='cifar10':
        train_set = np.array([es[0][i] for es in sio.loadmat(path)['gist'] for i in range(5500, es[0].shape[0])])
        query_set = np.array([es[0][i] for es in sio.loadmat(path)['gist'] for i in range(5400, 5500)])
        database = np.array([es[0][i] for es in sio.loadmat(path)['gist'] for i in range(0, 5400)])
        return query_set, database, train_set, Sun397(train_set), Sun397(database)
    elif name=='caltech101':
        data = np.array([es[0][i] for es in sio.loadmat(path)['gist'] for i in range(es[0].shape[0])])
        rng = np.random.RandomState(seed)
        ind_shuf = list(range(len(data)))
        rng.shuffle(ind_shuf)
        data = data[ind_shuf]
        query_len = round(len(data)*0.02)
        query = data[:query_len]
        database = data[query_len:]
        train_len = round(len(database)*0.1)
        train = database[:train_len]
        # print(data[0])
        return query, database[train_len:], Sun397(train), Sun397(database[train_len:]) 
    elif name=='halfdome':
        data = sio.loadmat(path)['data']
        data = data[~np.isnan(data).any(axis=1), :]
        rng = np.random.RandomState(seed)
        ind_shuf = list(range(len(data)))
        rng.shuffle(ind_shuf)
        data = data[ind_shuf]
        query_len = round(len(data)*0.02)
        query = data[:query_len]

        base = data[query_len:]
        train_len = round(len(base)*0.1)
        train = base[:train_len]
        # print(data[0])
        database = base[train_len:]
        return query, database, train, Sun397(train), Sun397(database)

    elif name=='echonest':
        query = np.load('dataset/EchoNest/queries.npy')
        train = np.load('dataset/EchoNest/EchoNest_user.npy')
        database = np.load('dataset/EchoNest/EchoNest_item.npy')
        return query, database, Sun397(train), Sun397(database)
    elif name=='lastfm':
        query = np.load('dataset/LastFM/queries_32D.npy')
        train = np.load('dataset/LastFM/LastFM_32D_user.npy')
        database = np.load('dataset/LastFM/LastFM_data32D.npy')
        return query[:len(query)//3], database[:len(database)//3], Sun397(train[:len(train)//3]), Sun397(database[:len(database)//3])

    elif name=='netflix':
        items = np.load('../dataset/Netflix_movies.npy')
        query = np.load('../dataset/Netflix_queries.npy')
        train_len = round(len(items)*0.02)
        train = items[:train_len]
        database = items[train_len:]

        length = int(len(query)/5)
        # print(length)
        query_valid = query[:length]
        queries = query[length:] # data[0:train_len]
        
        return queries, database, train, query_valid, Sun397(train), Sun397(database)
    elif name=='amazon':
        query = np.load('../dataset/amazon/amazon_query.npy')
        train = np.load('../dataset/amazon/amazon_train.npy')
        train = train[:int(len(train)/3)]
        database = np.load('../dataset/amazon/amazon_database.npy')
        
        length = int(len(query)/12)
        query_valid = query[:length]
        queries = query[length:] # data[0:train_len] [length:]

        return queries, database, train, query_valid, Sun397(train), Sun397(database)
    # elif name == 'movielens20M':
    #     return MovieLens20MDataset(path)
    # elif name == 'criteo':
    #     return CriteoDataset(path)
    # elif name == 'avazu':
    #     return AvazuDataset(path)
    else:
        raise ValueError('unknown dataset name: ' + name)


def get_model(name, dataset, query_valid, M, K, T, device, metric, e, init_q):
    """
    Hyperparameters are empirically determined, not opitmized.
    """
    D = dataset.dims
    if name == 'multipq':
        return ProjKmeans(query_valid, M, K, D, T, device, metric, e, init_q)
    elif name == 'deeppq':
        return KmeansNN(query_valid, M, K, D, T, metric, e)
    elif name == 'lightrec':
        return LightRec(M, K, D, T, metric)
    elif name == 'diffpq':
        return DiffPQ(M, K, D, T, metric)
    else:
        raise ValueError('unknown model name: ' + name)



class EarlyStopper(object):

    def __init__(self, num_trials, save_path):
        self.num_trials = num_trials
        self.trial_counter = 0
        self.best_accuracy = 1e+10
        self.save_path = save_path

    def is_continuable(self, model, accuracy):
        if accuracy - self.best_accuracy < 0:
            self.best_accuracy = accuracy
            self.trial_counter = 0
            torch.save(model, self.save_path)
            return True
        elif self.trial_counter + 1 < self.num_trials:
            self.trial_counter += 1
            return True
        else:
            return False


def train(model, model_name, optimizer, data_loader, loss, device, e, log_interval=100):
    model.train()
    # tk0 = tqdm.tqdm(data_loader, smoothing=0, mininterval=0.1)
    
    for batch, (X_B, y_B) in enumerate(data_loader):
        X_B = X_B.to(device)
        if model_name=='multipq':
            X_r, X_p, X_r_m, X_p_m, centroid, code, ormR, tmp, y_hard, y_soft, output = model(X_B)
            # u, v = directloss(X_B, X_p, X_r, y_hard, y_soft, ormR, tmp, centroid, e) 
            # output += v * 1.2 + u
            centroid = centroid.reshape(-1, centroid.shape[2])
        elif model_name=='deeppq':
            X_r, X_p, X_r_m, X_p_m, centroid, label, output = model(X_B)
        elif model_name=='lightrec':
            X_r, X_p, X_r_m, X_p_m, centroid, label, output = model(X_B)
        elif model_name=='diffpq':
            X_r, X_p, X_r_m, X_p_m, centroid, label, output = model(X_B)
        else:
            raise ValueError('unknown model name: ' + model_name)

        # print(loss(X_B, X_p_m))
        
        # print(X_r.shape, X_p.shape, X_p_m.shape, X_B.shape)
        # output = loss(X_r, X_p) #+ 1.5 * loss(X_B, X_r_m) #+ loss(X_p_m, X_B) #+ 0.3 * diversity(X_p, device) 
        optimizer.zero_grad()
        output.backward() # retain_graph=True
        optimizer.step()
        # print(X_r_m)
    # X_r, X_p, X_r_m, X_p_m, _c, _l = model(X_B) 
    return centroid


def test(model, model_name, data_loader, D, M, device):
    model.eval()
    test_loss = 0
    loss = ScaNN(D, M, device)
    # tk0 = tqdm.tqdm(data_loader, smoothing=0, mininterval=0.1)
    with torch.no_grad():
        for X_B, y_B in data_loader:
            X_B = X_B.to(device)
            # X_r, X_p, X_r_m, X_p_m, centroid, label, _ = model(X_B)
            if model_name=='multipq':
                X_r, X_p, X_r_m, X_p_m, centroid, label, ormR, tmp, y_hard, y_soft, output = model(X_B)
            elif model_name=='deeppq':
                X_r, X_p, X_r_m, X_p_m, centroid, label, _ = model(X_B)
            elif model_name=='lightrec':
                X_r, X_p, X_r_m, X_p_m, centroid, label, _ = model(X_B)
            elif model_name=='diffpq':
                X_r, X_p, X_r_m, X_p_m, centroid, label, _ = model(X_B)
            else:
                raise ValueError('unknown model name: ' + model_name)

            if test_loss==0:
                test_code = label
            else:
                test_code = torch.cat((test_code, label), 0)
            test_loss += torch.square(X_B - X_r_m).sum().item()
            # test_loss += loss.score_aware_loss(X_p, X_r).item()
            # test_loss += (abs(X_B - X_r_m)).sum().item()
            # print(X_r_m, test_loss)
    return test_loss, test_code


def get_optimizer(name, model, learning_rate, weight_decay):
    if name == 'multipq':
        return torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        # optimizer = torch.optim.Adam([
        #     {'params': model.proj.tmp, 'lr': learning_rate},
        #     # {'params': model.km.weight, 'lr': learning_rate},
        #     {'params': model.km.center, 'lr': learning_rate},  # center
        #     {'params': model.weight, 'lr': 1e-5}
        #     ]) # weight
        # return optimizer
    elif name == 'deeppq':
        optimizer = torch.optim.Adam([
            {'params': model.center, 'lr': learning_rate},  # center
            {'params': model.weight, 'lr': 1e-5}]) # weight
        return optimizer
    elif name == 'lightrec':
        return torch.optim.Adam(model.parameters(), lr=learning_rate)
    elif name=='diffpq':
        return torch.optim.Adam(model.parameters(), lr=learning_rate)
    else:
        raise ValueError('unknown model name: ' + name)
    
    

def main(dataset_name,
         dataset_path,
         model_name,
         epoch,
         learning_rate,
         batch_size,
         weight_decay,
         device,
         save_dir,
         num_codebooks,
         num_clusters,
         tempreture,
         metric,
         e,
         gamma):
    device = torch.device(device)
    queries, database, train_, query_valid, dataset, test_dataset = get_dataset(dataset_name, dataset_path)
    M, K, T, D = num_codebooks, num_clusters, tempreture, dataset.dims
    # I = torch.eye(D).to(device)
    
    print(train_.shape)
    init_q = torch.linalg.qr(torch.tensor(train_.T), mode='complete').Q
    init_q = init_q.to(device)

    train_length = int(len(dataset) * 0.8)
    valid_length = len(dataset) - train_length #int(len(dataset) * 0.2)
    # test_length = len(dataset) - train_length - valid_length
    train_dataset, valid_dataset = torch.utils.data.random_split(
        dataset, (train_length, valid_length))
    # print(train_dataset)
    train_data_loader = DataLoader(train_dataset, batch_size=batch_size, num_workers=8, shuffle=True)
    valid_data_loader = DataLoader(valid_dataset, batch_size=batch_size, num_workers=8, shuffle=True)
    test_data_loader = DataLoader(test_dataset, batch_size=batch_size, num_workers=8, shuffle=False)
    # print(train_dataset.shape)
    # return 0
    loss = torch.nn.MSELoss()
    query_valid = torch.Tensor(query_valid).double().to(device)
    model = get_model(model_name, dataset, query_valid, M, K, T, device, metric, e, init_q).to(device)
    optimizer = get_optimizer(model_name, model, learning_rate, weight_decay)
    # early_stopper = EarlyStopper(num_trials=8, save_path=f'{save_dir}/{model_name}.pt')
    # early_stopper = EarlyStopper(num_trials=5, save_path=f'{save_dir}/{model_name}.pt')
    # optimizer = torch.optim.Adam(params=model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=gamma)
    for epoch_i in range(epoch):
        codewords = train(model, model_name, optimizer, train_data_loader, loss, device, e)
        
        # with open(tfilename, 'a') as tf:
        #     tf.write(str(epoch_i)+"\t"+str(loss1)+"\n")
        scheduler.step()
        train_loss, train_label = test(model, model_name, train_data_loader, D, M, device)
        valid_loss, valid_label = test(model, model_name, valid_data_loader, D, M, device)
        # test_loss,  test_label  = test(model, model_name, test_data_loader,  D, M, device)
        print(epoch_i, train_loss, valid_loss)#, test_loss)
        # if not early_stopper.is_continuable(model, valid_loss):
            # break
        # auc, loss = test(model, valid_data_loader, device, criterion)
        
    # torch.save(model, f'movielens-20m/{embed_dim}_model.pkl')
    # torch.save(model.state_dict(), f'movielens-20m/{embed_dim}_parameter.pkl')
    
    #测试模型
    test_loss, test_label = test(model, model_name, test_data_loader, D, M, device)
    # auc, loss = test(model, test_data_loader, device, criterion)
    # print(f'test auc: {auc}')
    # with open(filename, 'a') as f:
    #     f.write("test auc"+str(auc))

    # codewords, label = model.km.center
    # ground_truth = np.load("../dataset/amazon/true_neighbors_top512.npy")[:,0]
    # pre_database = codewords @ test_label
    # inner = queries @ pre_database
    # neighbors_matrix = np.argsort(inner, axis = 0)
    # recall_atN_(neighbors_matrix, ground_truth)

    

    codebooks = codewords.cpu().detach().numpy()
    label = test_label.squeeze(-1).cpu().detach().numpy()

    # # raq = SearchNeighbors_AQ(M = M, K = K, D = D, aq_codebooks = codebooks, aq_codes = label, metric="dot_product")
    # # ground_truth = raq.brute_force_search(test, queries, metric="l2_distance")

    metric="dot_product"
    raq = SearchNeighbors_AQ(M = M, K = K, D = D, aq_codebooks = codebooks, aq_codes = label, metric=metric)
    ground_truth = raq.brute_force_search(database, queries, metric=metric)
    # # print(ground_truth)
    # ground_truth = np.load("../dataset/amazon/true_neighbors_top512.npy")[:,0]
    # print(ground_truth.shape)
    raq.aq_recall(queries=queries, topk=100, ground_truth=ground_truth) 
    neighbors_matrix = raq.par_neighbors(queries=queries, topk=512, njobs=2)
    # print(neighbors_matrix.shape)
    recall_atN(neighbors_matrix, ground_truth)
    exit()

    # rpq = SearchNeighbors_PQ(M=M, Ks=K, D=D, pq_codebook = codebooks, pq_codes = label, metric=metric)
    # ground_truth = rpq.brute_force_search(database, queries, metric = metric) 
    # neighbors_matrix = rpq.neighbors(queries,topk = 512)
    # recall_atN(neighbors_matrix, ground_truth)

    # exit()
    # return test_label # test_center, test_label



if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_name', default='amazon', help='caltech101, halfdome, sun397, cifar10, amazon, netflix, echonest, lastfm, movielens1M, avazu, movielens20M ')
    parser.add_argument('--dataset_path', default='../dataset/netflix', help='netflix, sun397.mat, cifar10.mat, criteo/train.txt, avazu/train, or ml-1m/ratings.dat')
    parser.add_argument('--model_name', default='multipq', help='deeppq, multipq, lightrec, diffpq')
    parser.add_argument('--epoch', type=int, default=100)
    parser.add_argument('--learning_rate', type=float, default=1e-2)
    parser.add_argument('--gamma', type=float, default=0.9)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--weight_decay', type=float, default=1e-5)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--save_dir', default='log')
    parser.add_argument('--num_codebooks', type=int, default=8, help='M')
    parser.add_argument('--num_clusters', type=int, default=256, help='K')
    parser.add_argument('--tempreture', type=float, default=0.8, help='T, 0.5')
    parser.add_argument('--e', type=float, default=0.6, help='e, 0.45')
    parser.add_argument('--metric', default='dot_product', help='l2_distance, dot_product')
    args = parser.parse_args()
    main(args.dataset_name,
         args.dataset_path,
         args.model_name,
         args.epoch,
         args.learning_rate,
         args.batch_size,
         args.weight_decay,
         args.device,
         args.save_dir,
         args.num_codebooks,
         args.num_clusters,
         args.tempreture, 
         args.metric,
         args.e,
         args.gamma
         )

