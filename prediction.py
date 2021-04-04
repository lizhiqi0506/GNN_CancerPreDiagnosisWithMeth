#coding:utf-8

import sys
import argparse
import numpy as np
import pandas as pd
import torch
from torch_geometric.data import DataLoader
from torch_scatter import scatter_mean
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv, GatedGraphConv,TopKPooling
import torch.optim as optim
from .node_classify.createDataset_nodeclassify import *


"""
python prediction.py -mode [predict mode] -input [file] -output [file] -type [primary site]
"""

parser = argparse.ArgumentParser(description="Prediction")
parser.add_argument('--mode', '-m', help="Select the prediction mode:[tissue, node, case-normal]")
parser.add_argument('--input', '-i', help="""The input file:
                                             if the -m argument is "node" then the input should be a Illumina 450k .txt file;
                                             otherwise the input should be a .pt file created by createDataset.py""")
parser.add_argument('--output', '-o', help='The output file containing the result of prediction')
parser.add_argument('--type', '-t', help='If your -m argument is "case-normal", you should choose the primary site that the data belong to')
args = parser.parse_args()
primary_sites = ['bladder','brain','breast','bronchus and lung','cervix uteri','corpus uteri','kidney','liver and intrahepatic bile ducts','prostate gland','stomach','thyroid gland']

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

input_dim_n = 200
hidden_dim_n = 100
num_classes_n = 11

def main():
    if args.mode == 'node':
        predict_n()

class GNN_n(nn.Module):
    def __init__(self):  # 不用传数据参数
        super(GNN_n, self).__init__()
        self.conv1 = GCNConv(input_dim_n, hidden_dim_n)
        self.conv2 = GCNConv(hidden_dim_n, hidden_dim_n)  
        self.conv3 = GCNConv(hidden_dim_n, hidden_dim_n) 
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim_n, hidden_dim_n // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim_n // 2, hidden_dim_n // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim_n // 2, num_classes_n))
        #self.conv5 = GATConv(256,dataset.num_classes,dropout=0.6)

    def forward(self, data):  #
        
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr  
        #print(x.shape)
        x = self.conv1(x, edge_index, edge_attr)  
        x = F.relu(x)                 
        x = F.dropout(x, training=self.training)  

        #print(x.shape)
        x = self.conv2(x, edge_index, edge_attr)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)

        #print(x.shape)
        x = self.conv3(x, edge_index, edge_attr)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)

        #print(x.shape)
        x = self.mlp(x)
        
        print(x.shape)
        x = F.log_softmax(x, dim=1)  

        return x

def predict_n():
    SAMPLING_RATIO = 0.5
    basedir = './data/'
    meth_mat_selected_df = pd.read_csv(
        basedir + 'meth_matrix_selected_200.csv', index_col=0)
    sites_selected = meth_mat_selected_df['composite element ref'].values
    input_file = args.input
    input_df = pd.read_csv(input_file, sep='\t')[
        ['Composite Element REF', 'Beta_value']].fillna(0).set_index('Composite Element REF')
    selected_beta_value = input_df.loc[sites_selected, 'Beta_value'].values
    selected_beta_value = torch.tensor(selected_beta_value)

    model = GNN_n()
    model.load_state_dict(torch.load("model_nodeclassify.pt"))
    model.to(device)
    model.eval()

    #meth_mat_maxstd_20k_df = pd.read_csv(basedir+'meth_matrix_maxstd20k.csv',index_col=0)
    meth_sites = meth_mat_selected_df.values[:, 0]
    beta_values = meth_mat_selected_df.values[:, 1:].T
    g = createGraph_n(selected_beta_value, beta_values, SAMPLING_RATIO)
    g.to(device)
    out = model(g)
    _, pred = torch.max(out, dim=1)
    pred = pred[0]
    if args.output is not None:
        f = open(args.output,"w")
        f.write(primary_sites[pred])
        f.close()
    else:
        print(primary_sites[pred])

def createGraph_n(selected_beta_value,beta_values,ratio):
    EDGE_THRESHOLD = 0.8
       
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
    X[0] = selected_beta_value
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




if __name__ == '__main__':
    main()
