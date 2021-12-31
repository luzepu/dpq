import numpy as np
# import pandas as pd
import scipy.io as sio
from torch.functional import tensordot
import torch.utils.data
from torch.utils.data import DataLoader, Dataset


class Sun397(Dataset):
    """
    Sun397

    Data preparation
        includes about 108K images which come from 397 different scenes. 

    :param dataset_path: Sun397 dataset path

    Reference:
        https://grouplens.org/datasets/movielens
    """
    def __init__(self, data):
        # self.data = np.array([es[0][i] for es in sio.loadmat(path)['gist'] for i in range(es[0].shape[0])])  # 108754
        # self.data = torch.tensor(data).to(device)
        self.data = data
        self.dims = self.data.shape[1]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index) :
        return (self.data[index].astype('float64'), 0)



# class Sun397(Dataset):


#     def __init__(self, dataset_path, sep=',', engine='c', header='infer'):
#         self.data = np.array([es[0][i] for es in sio.loadmat(dataset_path)['gist'] for i in range(es[0].shape[0])])  # 108754
#         # data = pd.read_csv(dataset_path, sep=sep, engine=engine, header=header).to_numpy()[:, :3]
#         # self.items = data[:, :2].astype(np.int) - 1  # -1 because ID begins from 1
#         # self.targets = self.__preprocess_target(data[:, 2]).astype(np.float32)
#         self.dims = self.data.shape[1]
#         # self.user_field_idx = np.array((0, ), dtype=np.long)
#         # self.item_field_idx = np.array((1,), dtype=np.long)

#     def __len__(self):
#         return self.data.shape[0]

#     def __getitem__(self, index):
#         # return self.items[index], self.targets[index]
#         return (self.data[index].astype('float64'), 0)

    # def getdata(self):
    #     return self.data
    # def __preprocess_target(self, target):
    #     target[target <= 3] = 0
    #     target[target > 3] = 1
    #     return target


# class MovieLens1MDataset(MovieLens20MDataset):
#     """
#     MovieLens 1M Dataset

#     Data preparation
#         treat samples with a rating less than 3 as negative samples

#     :param dataset_path: MovieLens dataset path

#     Reference:
#         https://grouplens.org/datasets/movielens
#     """

#     def __init__(self, dataset_path):
#         super().__init__(dataset_path, sep='::', engine='python', header=None)


# dataset = Sun397('sun397.mat')
# # print(data[0])

# train_length = int(len(dataset) * 0.8)
# valid_length = int(len(dataset) * 0.1)
# test_length = len(dataset) - train_length - valid_length
# train_dataset, valid_dataset, test_dataset = torch.utils.data.random_split(
#     dataset, (train_length, valid_length, test_length))

# train_data_loader = DataLoader(train_dataset, batch_size=64, num_workers=8)
# valid_data_loader = DataLoader(valid_dataset, batch_size=64, num_workers=8)
# test_data_loader = DataLoader(test_dataset, batch_size=64, num_workers=8)

# print(train_data_loader)