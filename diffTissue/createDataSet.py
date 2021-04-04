# %%
#!/usr/bin/python
# coding:utf-8

"""
作者：lzq
功能：构建差异共表达网络数据集_11类
版本：1
日期：2021-03-24

"""
import random
import os.path as osp
import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Data, InMemoryDataset, Dataset


class MyInMemoryDataset(InMemoryDataset):
    def __init__(self, root, transform=None, pre_transform=None):
        super(MyInMemoryDataset, self).__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return ['meth_matrix_maxstd_2k.csv']

    @property
    def processed_file_names(self):
        return ['dataset_wgcna_2k.pt']

    def download(self):
        # Download to `self.raw_dir`.
        return

    def process(self):
        # Read data into huge `Data` list.
        basedir = '/lustre/home/acct-clsdqw/clsdqw-jiangxue/lzq/gnn_nosex/data/'
        meth_mat_maxstd_2k_df = pd.read_csv(
            basedir + self.raw_file_names[0], index_col=0)
        #meth_mat_maxstd_20k_df = pd.read_csv(basedir+'meth_matrix_maxstd_500.csv',index_col=0)
        meth_sites = meth_mat_maxstd_2k_df.values[:, 0]
        beta_values = meth_mat_maxstd_2k_df.values[:, 1:].T

        data_list = []
        sampling_datas = sampling(beta_values)
        n_type, n_repeat, n_site, n_sample = sampling_datas.shape
        print("sampling data shape:",sampling_datas.shape)

        
        edge_threshold = 0.001
        powers = np.array([9,2,1,7,1,12,9,8,14,16,4])

        for i in range(n_type):
            #power, edge_threshold = powerAndThreshold(int(i))
            power = powers[i]
            for j in range(n_repeat):
                g = wgcnaNet(sampling_datas[i, j],power,int(i))
                data_list.append(g)

        print("num of graphs:", len(data_list))

        if self.pre_filter is not None:
            data_list = [
                data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        data, slice = self.collate(data_list)
        torch.save((data, slice), osp.join(self.processed_dir, 'dataset_wgcna_2k.pt'))


def sampling(beta_values):
    n_sample = 200
    n_repeat = 10

    def sampleFromData(mat):
        return sampleFromData_(mat, n_sample, n_repeat)

    # shape: (n_repeat,20k,n_sample)
    bladder = sampleFromData(beta_values[:438])
    brain = sampleFromData(beta_values[438:1124])
    breast = sampleFromData(beta_values[1124:2010])
    bronchus = sampleFromData(beta_values[2010:2924])
    cervix = sampleFromData(beta_values[2924:3233])
    #colon = sampleFromData(beta_values[3233:3576])
    corpus = sampleFromData(beta_values[3233:3707])
    kidney = sampleFromData(beta_values[3707:4579])
    liver = sampleFromData(beta_values[4579:5047])
    prostate = sampleFromData(beta_values[5047:5589])
    stomach = sampleFromData(beta_values[5589:5987])
    thyroid = sampleFromData(beta_values[5987:])

    datas = np.array([bladder, brain, breast, bronchus,cervix,corpus,kidney,liver,prostate,stomach,thyroid])     # shape: (11,n_repeat,20k,n_sample)
    return datas


def sampleFromData_(mat, n_sample, n_repeat):
    # 从一类cancer的样本中有放回的抽取n_sample个样本
    # 重复n_repeat次
    data = []
    for i in range(n_repeat):
        choice_index = np.random.choice(len(mat), n_sample)
        data.append(mat[choice_index].T.tolist())
    return np.array(data)


def powerAndThreshold(y):

    power_thr = np.array([[2, 0.7], [2, 0.82],[1,0.53],[2,0.8],[1,0.5],\
        [2, 0.85], [1, 0.45],[2,0.85],[2,0.7],[1,0.7],[2,0.82],[2,0.8]])  
    power, threshold = power_thr[y][0], power_thr[y][1]
    return power, threshold

def wgcnaNet(X, power, y):
    """
        X: 一类癌症样本
        power: WGCNA软阈值
        edge_threshold: 边阈值
        y: 这类样本的标签值
    """
    edge_threshold = 0.001
    # 节点间的相关系数矩阵
    cor_mat = np.corrcoef(X)    # shape:(n_nodes,n_nodes)
    # 邻接矩阵
    A = adjMatrix(cor_mat, power, edge_threshold)
    # 边索引
    edge_index = edgeIndex(A)
    edge_attr = edgeAttr(edge_index, cor_mat, power)
    X = torch.tensor(X, dtype=torch.float)
    y = torch.tensor([y], dtype=torch.long)
    g = Data(x=X, edge_index=edge_index, edge_attr=edge_attr,  y=y)
    print(g)
    return g


# > powers
# [1]  9  2  1  7  1  1 12  9  8 14 16  4

def adjMatrix(cor_mat, power, threshold):
    # 由相关系数矩阵构建自邻接矩阵
    A = cor_mat
    r, c = A.shape
    edge_attr = []
    for i in range(r):
        for j in range(c):
            if pow(A[i][j], power) > threshold:
                A[i][j] = 1
            else:
                A[i][j] = 0
    return A

def getEdgeNum(mat,power,threshold):
    cor_mat = np.corrcoef(mat)
    A = adjMatrix(cor_mat,power,threshold)
    edge_index = edgeIndex(A)
    return len(edge_index[0])


def edgeIndex(A):
    index1 = []
    index2 = []
    n = len(A)
    for i in list(range(n)):
        for j in list(range(n)):
            if i != j and A[i][j] == 1:
                index1.append(i)
                index2.append(j)
    index = torch.tensor([index1, index2], dtype=torch.long)
    return index


def edgeAttr(edge_index, cor_mat, power):
    edge_attr = []
    index1 = edge_index[0]
    index2 = edge_index[1]
    n = len(index1)
    for i in range(n):
        ind1 = index1[i]
        ind2 = index2[i]
        edge_attr.append(pow(cor_mat[ind1,ind2],power))

    edge_attr = torch.tensor(edge_attr, dtype=torch.float)
    return edge_attr


def degMatrix(A):
    # 由邻接矩阵构建度矩阵
    D = np.zeros(A.shape, dtype=np.int)
    n = len(A)
    for i in range(n):
        D[i, i] = np.sum(A[i])
    return D


def main():
    dataset = MyInMemoryDataset(root='/lustre/home/acct-clsdqw/clsdqw-jiangxue/lzq/gnn_nosex/test/')
    

#%%
if __name__ == '__main__':
    main()

# %%
