import csv
import itertools
import random

import numpy as np
import torch
import torch.utils.data as Data
from transformers import AutoTokenizer
import fasttext
from faker import Faker


class TestDataset(Data.Dataset):

    def __init__(self,
                 test_path_mat, #test set矩阵的路径，矩阵是提前将text通过encoder得到的vectors
                 da,
                 plm = 'fasttext'
                 ):

        self.plm = plm
        self.da = da
        mats = np.load(test_path_mat, allow_pickle=True)
        self.data_mats = mats # self.data_mats[k] is a matrix of the kth col
        self.data_size = len(self.data_mats)  # of cols in total

    def __len__(self):
        """Return the size of the dataset."""
        return self.data_size

    def __getitem__(self, index):
        """Return a item of the dataset.
        """
        x = self.data_mats[index]
        return x


    @staticmethod
    def pad(batch):
        """Merge the embed_mats of different cols into a big mat
        Args:
            batch (list of arrays): a list of arrays, eac array is respect to a col
        Returns:
            LongTensor: a big mat
            LongTensor: index, which indicates that each col contains ? cells
        """

        mat = batch
        index_mat = []
        concat_array = np.concatenate(mat, axis=0)
        for arr in mat:
            index_mat.append(arr.shape[0])
        return torch.tensor(concat_array, dtype=torch.float32), torch.tensor(index_mat)

class MyDataset(Data.Dataset):

    def __init__(self,
                 anchor_path,
                 auglist_path,
                 list_size,
                 training = 'true',
                 plm = 'fasttext'
                 ):

        query = np.load(anchor_path, allow_pickle=True)
        # query = query
        self.anchor_mats = query
        self.data_size = len(self.anchor_mats)

        item = np.load(auglist_path, allow_pickle=True)
        item = item
        item_list = []

        for i in range(0, len(item), list_size):
            sub_list = item[i:i + list_size].tolist()
            item_list.append(sub_list)
        self.item_mats = item_list
        indicies = [k * list_size for k in range(len(query))]
        pos = item[indicies]
        self.pos_mats = pos
        self.plm = plm
        self.training = training
        self.list_size = list_size


    def __len__(self):
        """Return the size of the dataset."""
        return self.data_size

    def __getitem__(self, index):
        """Return a item of the dataset.
        """
        query = self.anchor_mats[index]
        item_list = self.item_mats[index]
        pos = self.pos_mats[index]
        index_list = []
        for itme in item_list:
            index_list.append(itme.shape[0])
        return query, pos, item_list, index_list

    @staticmethod
    def pad(batch):
        query, pos, item_list, index_list  = zip(*batch)
        # query, pos, item_list, index_list, label = zip(*batch)
        query_index = []
        pos_index = []
        concat_query = np.concatenate(query, axis=0)
        concat_pos = np.concatenate(pos, axis=0)
        # array_generator = itertools.chain.from_iterable(item_list)
        all_lists = [item for sublist in item_list for item in sublist]
        concat_item = np.concatenate(all_lists, axis=0)
        merged_list = list(itertools.chain(*index_list))
        for arr in query:
            query_index.append(arr.shape[0])
        for arr in pos:
            pos_index.append(arr.shape[0])
        return torch.tensor(concat_query, dtype=torch.float32), torch.tensor(concat_pos, dtype=torch.float32), torch.tensor(concat_item, dtype=torch.float32),\
               torch.tensor(query_index), torch.tensor(pos_index), torch.tensor(merged_list)
            # , torch.tensor(label)

class RankDataset(Data.Dataset):

    def __init__(self,
                 anchor_path,
                 auglist_path,
                 list_size,
                 training = 'true',
                 plm = 'fasttext'
                 ):
        max = 0

        query = np.load(anchor_path, allow_pickle=True)
        self.anchor_mats = query

        item = np.load(auglist_path, allow_pickle=True)
        item_list = []

        for i in range(0, len(item), list_size):
            sub_list = item[i:i + list_size].tolist()
            item_list.append(sub_list)
        self.item_mats = item_list
        self.plm = plm
        self.training = training
        self.data_size = len(self.anchor_mats)
        self.list_size = list_size
        # self.y = y

    # def augemt(self): #todo

    def __len__(self):
        """Return the size of the dataset."""
        return self.data_size

    def __getitem__(self, index):
        """Return a item of the dataset.
        """
        query = self.anchor_mats[index]
        item_list = self.item_mats[index]
        index_list = []
        # label = self.y[index]
        for itme in item_list:
            index_list.append(itme.shape[0])

        return query, item_list, index_list


    @staticmethod
    def pad(batch):
        query, item_list, index_list = zip(*batch)
        query_index = []
        concat_query = np.concatenate(query, axis=0)
        # array_generator = itertools.chain.from_iterable(item_list)
        all_lists = [item for sublist in item_list for item in sublist]
        concat_item = np.concatenate(all_lists, axis=0)
        merged_list = list(itertools.chain(*index_list))
        for arr in query:
            query_index.append(arr.shape[0])
        # label = np.vstack(label)
        return torch.tensor(concat_query, dtype=torch.float32, requires_grad=True), torch.tensor(concat_item, dtype=torch.float32, requires_grad=True), torch.tensor(query_index), torch.tensor(merged_list)\
            # , torch.tensor(label)
