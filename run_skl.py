import numpy as np
import scipy as sp
import scipy.io as sio
from sklearn.cluster import KMeans
from model import mkmeansnn
from multipq import mdkmeansnn
from lightrec import lightrecnn
from kdencoding import kdencodingnn
from opq import ProductQuantization, OptimizedProductQuantization
from DeepPQ import DeepProgressiveQuantization
from SPQ3 import SoftPQuantization
from DiffPQ import DiffPQuantization
from diverse_direct import directPQ
# from diffdiversedirect import D3PQ
# from tripleDPQ import D3PQ
from skewkmeans import sD3PQ
from dotmultipq import dotmultipq
import random
import torch
from evaluationRecall import SearchNeighbors_PQ, SearchNeighbors_AQ, recall_atN # Recall_AQ
# from recall import recall

def main(device):
    device = torch.device(device)
    # dataset = get_dataset(dataset_name, dataset_path)
    seed = 10
    random.seed(seed)
    np.random.seed(seed)

    # data = np.array([es[0][i] for es in sio.loadmat(r'./dataset/sun397.mat')['gist'] for i in range(es[0].shape[0])])  # 108754
    # rng = np.random.RandomState(seed)
    # ind_shuf = list(range(len(data)))
    # rng.shuffle(ind_shuf)
    # data = data[ind_shuf]

    # query_len = round(len(data)*0.02)
    # query = data[:query_len]

    # database = data[query_len:]
    # train_len = round(len(database)*0.1)
    # train = database[:train_len]    

    # valid = query
    # test = database[train_len:]

    # path = '../dataset/cifar10.mat'
    # train = np.array([es[0][i] for es in sio.loadmat(path)['gist'] for i in range(5500, es[0].shape[0])])
    # query = np.array([es[0][i] for es in sio.loadmat(path)['gist'] for i in range(5400, 5500)])
    # database = np.array([es[0][i] for es in sio.loadmat(path)['gist'] for i in range(0, 5400)])

    # print(data[0])
    # train_len = round(len(data)*0.15)
    # # print(train_len)
    # train = data[:train_len]
    # test = data[train_len:train_len*2]
    # valid = data[train_len*2:train_len*3]
    # dim = train.shape[1] # 512
    
    # print(codebooks.shape)
    # print(label)
    # filename1 = f'./codebooks/multipq-8_128.npy'
    # np.save(filename1, codebooks, allow_pickle=True, fix_imports=True)
    # filename2 = f'./codes/multipq-8_128.npy'
    # np.save(filename2, label, allow_pickle=True, fix_imports=True)   


#multipq
    # num_clusters = 128
    # M, K, D = 8, num_clusters, dim
    # X = train # data[train_len:]
    # queries = valid # data[0:train_len]
    # codewords, label = mdkmeansnn(train, test, num_clusters, device, M)
    # codewords = codewords.cpu().detach().numpy()
    # label = label.squeeze(-1).cpu().detach().numpy()
    
    # codebooks = codewords[0, :, :]
    # for i in range(1,M):
    #     centroid = codewords[i, :, :]
    #     codebooks = np.r_[codebooks,centroid]
           
    # # raq = SearchNeighbors_AQ(M = M, K = K, D = D, aq_codebooks = codebooks, aq_codes = label, metric="dot_product")
    # # ground_truth = raq.brute_force_search(test, queries, metric="dot_product")
    # raq = SearchNeighbors_AQ(M = M, K = K, D = D, aq_codebooks = codebooks, aq_codes = label, metric="l2_distance")
    # ground_truth = raq.brute_force_search(test,queries,metric="l2_distance")
    # raq.aq_recall(queries=queries, topk=100, ground_truth=ground_truth)  
    # neighbors_matrix = raq.par_neighbors(queries=queries, topk=512, njobs=4)
    # recall_atN(neighbors_matrix,ground_truth)
    # exit()


#diversedirectpq
    # num_clusters = 10
    # M, K, D = 2, num_clusters, dim
    # X = train # data[train_len:]
    # queries = valid # data[0:train_len]
    # codewords, label = directPQ(train, test, num_clusters, device, M)
    # codewords = codewords.cpu().detach().numpy()
    # label = label.squeeze(-1).cpu().detach().numpy()
    # codebooks = codewords[0, :, :]
    # for i in range(1,M):
    #     centroid = codewords[i, :, :]
    #     codebooks = np.r_[codebooks,centroid]
    # raq = SearchNeighbors_AQ(M = M, K = K, D = D, aq_codebooks = codebooks, aq_codes = label, metric="l2_distance")
    # ground_truth = raq.brute_force_search(X,queries,metric="l2_distance")
    # raq.aq_recall(queries=queries, topk=100, ground_truth=ground_truth)  
    # neighbors_matrix = raq.par_neighbors(queries=queries, topk=512, njobs=4)
    # recall_atN(neighbors_matrix,ground_truth)
    # exit()

# D3PQ
    # num_clusters = 128
    # M, K, D = 8, num_clusters, dim
    # X = train # data[train_len:]
    # queries = valid # data[0:train_len]
    # codewords, label = D3PQ(train, test, num_clusters, device, M)
    # codewords = codewords.cpu().detach().numpy()
    # label = label.squeeze(-1).cpu().detach().numpy()
    # codebooks = codewords[0, :, :]
    # for i in range(1,M):
    #     centroid = codewords[i, :, :]
    #     codebooks = np.r_[codebooks,centroid]
    # raq = SearchNeighbors_AQ(M = M, K = K, D = D, aq_codebooks = codebooks, aq_codes = label, metric="l2_distance")
    # ground_truth = raq.brute_force_search(test, queries, metric="l2_distance")
    # raq.aq_recall(queries=queries, topk=100, ground_truth=ground_truth)  
    # neighbors_matrix = raq.par_neighbors(queries=queries, topk=512, njobs=4)
    # recall_atN(neighbors_matrix,ground_truth)
    # exit()

# dotmultipq
    # num_clusters = 128
    # M, K, D = 8, num_clusters, dim
    # X = train # data[train_len:]
    # queries = valid # data[0:train_len]
    # codewords, label = dotmultipq(train, test, num_clusters, device, M)
    # codewords = codewords.cpu().detach().numpy()
    # label = label.squeeze(-1).cpu().detach().numpy()
    # codebooks = codewords[0, :, :]
    # for i in range(1,M):
    #     centroid = codewords[i, :, :]
    #     codebooks = np.r_[codebooks,centroid]
    # raq = SearchNeighbors_AQ(M = M, K = K, D = D, aq_codebooks = codebooks, aq_codes = label, metric="dot_product")
    # ground_truth = raq.brute_force_search(test, queries, metric="dot_product")
    # raq.aq_recall(queries=queries, topk=100, ground_truth=ground_truth)  
    # # neighbors_matrix = raq.par_neighbors(queries=queries, topk=512, njobs=4)
    # # recall_atN(neighbors_matrix,ground_truth)
    # exit()


# skew+kmeans
    # num_clusters = 128
    # M, K, D = 8, num_clusters, dim
    # X = train # data[train_len:]
    # queries = valid # data[0:train_len]
    # codewords, label = sD3PQ(train, test, num_clusters, device, M)
    # codewords = codewords.cpu().detach().numpy()
    # label = label.squeeze(-1).cpu().detach().numpy()
    # codebooks = codewords[0, :, :]
    # for i in range(1,M):
    #     centroid = codewords[i, :, :]
    #     codebooks = np.r_[codebooks,centroid]
    # raq = SearchNeighbors_AQ(M = M, K = K, D = D, aq_codebooks = codebooks, aq_codes = label, metric="l2_distance")
    # ground_truth = raq.brute_force_search(X,queries,metric="l2_distance")
    # raq.aq_recall(queries=queries, topk=100, ground_truth=ground_truth)  
    # neighbors_matrix = raq.par_neighbors(queries=queries, topk=512, njobs=4)
    # recall_atN(neighbors_matrix,ground_truth)
    # exit()

#lightrecnn
    # num_clusters = 128
    # M, K, D = 8, num_clusters, dim
    # X = train # data[train_len:]
    # queries = test # data[0:train_len]
    # codebooks, codes = lightrecnn(train, test, num_clusters, M, device)
    # codebooks = codebooks.cpu().detach().numpy()
    # codes = codes.cpu().detach().numpy()

    # # recall(codebooks, codes, X, queries, M, K, D)
    # raq = Recall_AQ(M = M, K = K, D = D, aq_codebooks = codebooks, aq_codes = codes, metric="dot_product") #l2_distance
    # ground_truth = raq.brute_force_search(X, queries,metric="dot_product")
    # raq.aq_recall(queries=queries, topk=100, ground_truth=ground_truth)  
    # exit()


# DeepProgressiveQuantization
    # num_clusters = 256
    # M, K, D = 4, num_clusters, dim
    # X = train # data[train_len:]
    # queries = valid # data[0:train_len]
    # codebooks, codes = DeepProgressiveQuantization(train, test, num_clusters, M, device)
    # codebooks = codebooks.cpu().detach().numpy()
    # codes = codes.cpu().detach().numpy()
    # # raq = Recall_AQ(M = M, K = K, D = D, aq_codebooks = codebooks, aq_codes = codes, metric="dot_product")
    # # ground_truth = raq.brute_force_search(X, queries,metric="dot_product")
    # # raq.aq_recall(queries=queries, topk=100, ground_truth=ground_truth)  
    # # exit()
    # raq = SearchNeighbors_AQ(M = M, K = K, D = D, aq_codebooks = codebooks, aq_codes = codes, metric="l2_distance")
    # ground_truth = raq.brute_force_search(test, queries, metric="l2_distance")
    # # raq = SearchNeighbors_AQ(M = M, K = K, D = D, aq_codebooks = codebooks, aq_codes = codes, metric="dot_product")
    # # ground_truth = raq.brute_force_search(test, queries, metric="dot_product")
    # raq.aq_recall(queries=queries, topk=100, ground_truth=ground_truth)  
    # neighbors_matrix = raq.par_neighbors(queries=queries, topk=512, njobs=4)
    # recall_atN(neighbors_matrix,ground_truth)
    # exit()

# SoftPQuantization
    # num_clusters = 256
    # alpha = 20
    # M, K, D = 256, num_clusters, dim
    # X = train # data[train_len:]
    # queries = valid # data[0:train_len]
    # codebooks, codes = SoftPQuantization(train, test, M, num_clusters, alpha, device)
    # codebooks = codebooks.cpu().detach().numpy()
    # codes = codes.cpu().detach().numpy()
    # # raq = Recall_AQ(M = 1, K = K, D = D, aq_codebooks = codebooks, aq_codes = codes, metric="l2_distance")
    # # ground_truth = raq.brute_force_search(X, queries,metric="l2_distance")
    # # raq.aq_recall(queries=queries, topk=100, ground_truth=ground_truth)
    # # X = np.reshape(X, (X.shape[0], M, D // M)).permute(1,0,2)  
    # # queries = np.reshape(queries, (queries.shape[0], M, D // M)).permute(1,0,2) 
    # rpq = Recall_PQ(M=M, Ks=K, D=D, pq_codebook = codebooks, pq_codes = codes, metric="l2_distance")
    # ground_truth = rpq.brute_force_search(X, queries, metric = "l2_distance") 
    # rpq.pq_recall(queries=queries, topk=100, ground_truth=ground_truth)
    # exit()

# DiffPQ
    # num_clusters = 128
    # M, Ks, D = 8, num_clusters, dim
    # X = train # data[train_len:]
    # queries = query # data[0:train_len]
    # codewords, label= DiffPQuantization(train, database, num_clusters, device, M)
    # codewords = codewords.cpu().detach().numpy()
    # label = label.squeeze(-1).cpu().detach().numpy()
    # rpq = SearchNeighbors_PQ(M=M, Ks=Ks, D=D, pq_codebook = codewords, pq_codes = label, metric="l2_distance")
    # ground_truth = rpq.brute_force_search(database, queries, metric = "l2_distance") 
    # neighbors_matrix = rpq.neighbors(queries,topk = 512)
    # recall_atN(neighbors_matrix, ground_truth)
    # exit()

# kdencoding
    # num_clusters = 128
    # M, K, D = 8, num_clusters, dim
    # X = train # data[train_len:]
    # queries = test # data[0:train_len]
    # codewords, label, H = kdencodingnn(train, test, num_clusters, device, M)
    # codewords = codewords.cpu().detach().numpy()
    # label = label.squeeze(-1).cpu().detach().numpy()
    # H = H.cpu().detach().numpy()
    # # print(codewords.shape, label.shape)
    # codebooks = codewords[0, :, :]
    # for i in range(1,M):
    #     centroid = codewords[i, :, :]
    #     codebooks = np.r_[codebooks,centroid]
    # # print(codebooks.shape, label.shape)
    # codebooks = np.matmul(codebooks, H)
    # raq = Recall_AQ(M = M, K = K, D = D, aq_codebooks = codebooks, aq_codes = label, metric="l2_distance")
    # ground_truth = raq.brute_force_search(X,queries,metric="l2_distance")
    # raq.aq_recall(queries=queries, topk=100, ground_truth=ground_truth)  
    # exit()



    # km_ = KMeans(num_clusters)
    # km_.fit(train)
    # score = km_.score(train)
    # print("kmeans score:",score)
    # exit()

    # data = np.array([es[0][i] for es in sio.loadmat("dataset/caltech101.mat")['gist'] for i in range(es[0].shape[0])])
    # rng = np.random.RandomState(seed)
    # ind_shuf = list(range(len(data)))
    # rng.shuffle(ind_shuf)
    # data = data[ind_shuf]

    # query_len = round(len(data)*0.02)
    # query = data[:query_len]

    # database = data[query_len:]
    # train_len = round(len(database)*0.1)
    # train = database[:train_len]
    # # print(data[0])
    # # query, database[train_len:], Sun397(train), Sun397(database[train_len:]) 
    # database = database[train_len:]
    # train_set = np.array([es[0][i] for es in sio.loadmat(path)['gist'] for i in range(5500, es[0].shape[0])])
    # query_set = np.array([es[0][i] for es in sio.loadmat(path)['gist'] for i in range(5400, 5500)])
    # database = np.array([es[0][i] for es in sio.loadmat(path)['gist'] for i in range(0, 5400)])
    # data = np.array([es[0][i] for es in sio.loadmat(path)['gist'] for i in range(es[0].shape[0])])  # 108754

    # rng = np.random.RandomState(seed)
    # ind_shuf = list(range(len(data)))
    # rng.shuffle(ind_shuf)
    # data = data[ind_shuf]

    # query_len = round(len(data)*0.02)
    # query = data[:query_len]

    # database = data[query_len:]
    # train_len = round(len(database)*0.1)
    # train = database[:train_len]
    # # print(data[0])
    # database = database[train_len:]
    # return query, database[train_len:], Sun397(train), Sun397(database[train_len:])
    # items = np.load('../dataset/Netflix_movies.npy')
    # query = np.load('../dataset/Netflix_queries.npy')
    # train_len = round(len(items)*0.02)
    # train = items[:train_len]
    # database = items[train_len:]   
    
    # length = int(len(query)/3)
    # # print(length)
    # query_valid = query[:length]
    # query = query[length:] 
    # query = np.load('dataset/LastFM/queries_32D.npy')
    # train = np.load('dataset/LastFM/LastFM_32D_user.npy')
    # database = np.load('dataset/LastFM/LastFM_data32D.npy')
    # # return , database[:len(database)//3], Sun397(train[:len(train)//3]), Sun397(database[:len(database)//3])
    # query = query[:len(query)//3]
    # database = database[:len(database)//3]
    # train = train[:len(train)//3]
    # queries = np.load('dataset/LastFM/queries_32D.npy')
    # train = np.load('dataset/LastFM/LastFM_32D_user.npy')
    # database = np.load('dataset/LastFM/LastFM_data32D.npy')
    # path = '../dataset/halfdome.mat'

    # data = sio.loadmat(path)['data']
    # # print(data.shape)
    # data = data[~np.isnan(data).any(axis=1), :]
    # rng = np.random.RandomState(seed)
    # ind_shuf = list(range(len(data)))
    # rng.shuffle(ind_shuf)
    # data = data[ind_shuf]
    # query_len = round(len(data)*0.02)
    # query = data[:query_len]
    
    query = np.load('../dataset/amazon/amazon_query.npy')
    train = np.load('../dataset/amazon/amazon_train.npy')
    train = train[:int(len(train)/3)]
    database = np.load('../dataset/amazon/amazon_database.npy')
    
    length = int(len(query)/12)
    query_valid = query[:length]
    queries = query[length:] # data[0:train_len] [length:]    
    # # print(data.shape)
    # # exit()
    # base = data[query_len:]
    # train_len = round(len(base)*0.1)
    # train = base[:train_len]
    # # print(data[0])
    # database = base[train_len:]
    dim = train.shape[1]
    # print(dim, queries.shape, database.shape)
#pq
    # random.seed(seed)
    # np.random.seed(seed)
    # num_clusters = 128
    # M, Ks, D = 8, num_clusters, dim
    # X = train # data[train_len:]
    # queries = query # data[0:train_len]
    # test = database
    # pq = ProductQuantization(M, num_clusters, dim)
    # codewords = pq.fit(train)
    # _, score, label = pq.predict(test)
    # print("pq test score:", score)
    # # X_code, score ,label = pq.predict(X)
    # print("pq train score:", score)
    # label = label.reshape(-1,M)
    # # rpq = Recall_PQ(M=M, Ks=Ks, D=D, pq_codebook = codewords, pq_codes = label, metric="l2_distance")
    # # ground_truth = rpq.brute_force_search(X, queries, metric = "l2_distance") 
    # # rpq.pq_recall(queries=queries, topk=100, ground_truth=ground_truth)
    # metric = "l2_distance" #"dot_product"
    # rpq = SearchNeighbors_PQ(M=M, Ks=Ks, D=D, pq_codebook = codewords, pq_codes = label, metric=metric)
    # # print()
    # ground_truth = rpq.brute_force_search(test, queries, metric=metric) 
    # neighbors_matrix = rpq.neighbors(queries,topk = 512)
    # recall_atN(neighbors_matrix, ground_truth)
    # exit()


#opq
    random.seed(seed)
    np.random.seed(seed)
    num_clusters = 256
    M, Ks, D = 8, num_clusters, dim
    X = train # data[train_len:]
    test = database
    queries = query # data[0:train_len]
    opq = OptimizedProductQuantization(M, num_clusters, dim)
    R, codewords = opq.fit(train, test)
    X_code, score, label = opq.predict(test)
    # print(codewords.shape, X_code.shape, score, label.shape)
    # label = label.reshape(-1,M)
    # X = X @ R
    # queries = queries @ R
    metric = "dot_product" # "l2_distance" #"dot_product"
    rpq = SearchNeighbors_PQ(M=M, Ks=Ks, D=D, pq_codebook = codewords, pq_codes = label, metric=metric)
    ground_truth = rpq.brute_force_search(test@ R, queries@ R, metric = metric) 
    neighbors_matrix = rpq.neighbors(queries @ R,topk = 512)
    recall_atN(neighbors_matrix, ground_truth)

    rpq.pq_recall(queries=queries, topk=100, ground_truth=ground_truth)
    _, score, label = opq.predict(test)
    print("opq score:",score)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    # parser.add_argument('--dataset_name', default='movielens20M', help='movielens1M, avazu, movielens20M ')
    # parser.add_argument('--dataset_path', default='ml-20m/ratings.csv', help='criteo/train.txt, avazu/train, or ml-1m/ratings.dat')
    # parser.add_argument('--model_name', default='fm')
    # parser.add_argument('--epoch', type=int, default=100)
    # parser.add_argument('--learning_rate', type=float, default=0.001)
    # parser.add_argument('--batch_size', type=int, default=2048)
    # parser.add_argument('--weight_decay', type=float, default=1e-6)
    parser.add_argument('--device', default='cuda:1') # cuda:8 cpu
    # parser.add_argument('--save_dir', default='chkpt')
    args = parser.parse_args(args=[])
    main(args.device)
