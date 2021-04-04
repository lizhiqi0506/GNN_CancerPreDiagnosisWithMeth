#%%
#!/usr/bin/python
#coding:utf-8

"""
作者：lzq
功能：构建图数据集
日期：2021-04-03
"""
import random
import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Data, InMemoryDataset, DataLoader

EDGE_THRESHOLD = 0.8
SAMPLING_RATIO = 0.5
REPEAT = 30


class NodeClassifyDataset(InMemoryDataset):
    def __init__(self, root, transform=None, pre_transform=None):
        super(NodeClassifyDataset, self).__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return ['meth_matrix_selected_200.csv']

    @property
    def processed_file_names(self):
        return ['dataset_30subg_edge80.pt']

    def download(self):
        # Download to `self.raw_dir`.
        return

    def process(self):
        # Read data into huge `Data` list.
        basedir = '/lustre/home/acct-clsdqw/clsdqw-jiangxue/lzq/gnn_nosex/data/'
        meth_mat_selected_df = pd.read_csv(basedir + self.raw_file_names[0],index_col=0)
        #meth_mat_maxstd_20k_df = pd.read_csv(basedir+'meth_matrix_maxstd20k.csv',index_col=0)
        meth_sites = meth_mat_selected_df.values[:,0]
        beta_values = meth_mat_selected_df.values[:,1:].T
        data_list = []
        for i in range(REPEAT):
            g = createGraph_N(beta_values,SAMPLING_RATIO)
            train_mask = torch.tensor(np.array(list(range(len(g.y))))<int(len(g.y)*0.7))
            test_mask = torch.tensor(np.array(list(range(len(g.y))))>=int(len(g.y)*0.3))
            g['train_mask'] = train_mask
            g['test_mask'] = test_mask
            data_list.append(g)

        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]
        print(self.processed_paths[0])
        data, slices = self.collate(data_list)
        torch.save((data, slices), "/lustre/home/acct-clsdqw/clsdqw-jiangxue/lzq/gnn_nosex/node_classify/processed/dataset_30subg_edge80.pt")


#%%
def createGraph(beta_values,ratio):
       
    def sampleFromData(mat):
        return sampleFromData_(mat, ratio)
    
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
    """
    bladder = beta_values[:438]
    brain = beta_values[438:1124]
    breast = beta_values[1124:2010]
    bronchus = beta_values[2010:2924]
    cervix = beta_values[2924:3233]
    colon = beta_values[3233:3576]
    corpus= beta_values[3576:4050]
    kidney = beta_values[4050:4922]
    liver = beta_values[4922:5390]
    prostate = beta_values[5390:5932]
    stomach = beta_values[5932:6330]
    thyroid = beta_values[6330:]
    """
    # 特征集
    X = np.concatenate((bladder,brain,breast,bronchus,cervix,corpus,kidney,liver,prostate,stomach,thyroid),axis=0)
    #X = beta_values.astype(np.float)
    # 节点数
    n_nodes = len(X)
    # 标签集
    Y = [0]*int(438*ratio) + [1]*int(686*ratio) + [2]*int(886*ratio) + [3]*int(914*ratio) + [4]*int(309*ratio) + [5]*int(474*ratio) + [6]*int(872*ratio) + [7]*int(468*ratio) + [8]*int(542*ratio) + [9]*int(398*ratio) + [10]*int(570*ratio)
    # 将X和Y随机排序
    state = np.random.get_state()
    np.random.shuffle(X)
    np.random.set_state(state)
    np.random.shuffle(Y)
    
    # 节点间的相关系数矩阵
    cor_mat = np.corrcoef(X)    # shape:(n_nodes,n_nodes)
    # 邻接矩阵
    A = adjMatrix(cor_mat,EDGE_THRESHOLD)
    # 度数矩阵
    D = degMatrix(A)
    # 边索引
    edge_index = edgeIndex(A)
    edge_attr = edgeAttr(edge_index, cor_mat, EDGE_THRESHOLD)

    X = torch.tensor(X,dtype=torch.float)
    Y = torch.tensor(Y,dtype=torch.long)
    g = Data(x=X,edge_index=edge_index,edge_attr=edge_attr,y=Y)

    return g


def sampleFromData_(mat,ratio):
    # 从一类cancer的样本中抽取一定比例的样本
    n_sample = int(len(mat)*ratio)
    mat = mat.tolist()
    mat_sample = np.array(random.sample(mat, n_sample))
    return mat_sample

def adjMatrix(cor_mat,threshold):
    # 由相关系数矩阵构建自邻接矩阵
    A = cor_mat
    r,c = A.shape
    for i in range(r):
        for j in range(c):
            if A[i][j] > threshold:
                A[i][j] = 1
            else:
                A[i][j] = 0
    return A

def edgeAttr(edge_index, cor_mat, edge_threshold):
    edge_attr = []
    index1 = edge_index[0]
    index2 = edge_index[1]
    n = len(index1)
    for i in range(n):
        ind1 = index1[i]
        ind2 = index2[i]
        edge_attr.append((cor_mat[ind1,ind2]-edge_threshold)/(1-edge_threshold))

    edge_attr = torch.tensor(edge_attr, dtype=torch.float)
    return edge_attr

def edgeIndex(A):
    index1 = []
    index2 = []
    n = len(A)
    for i in list(range(n)):
        for j in list(range(n)):
            if i != j and A[i][j] == 1:
                index1.append(i)
                index2.append(j)
    index = torch.tensor([index1,index2],dtype=torch.long)
    return index

def degMatrix(A):
    # 由邻接矩阵构建度矩阵
    D = np.zeros(A.shape,dtype=np.int)
    n = len(A)
    for i in range(n):
        D[i,i] = np.sum(A[i])
    return D
                


#%%
def main():
    dataset = MyOwnDataset(root='/lustre/home/acct-clsdqw/clsdqw-jiangxue/lzq/gnn_nosex/node_classify/')

if __name__ == '__main__':
    main()
