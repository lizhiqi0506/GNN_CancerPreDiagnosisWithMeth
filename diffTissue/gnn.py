#%%
#!/usr/bin/python
#coding:utf-8

"""
作者：lzq
功能：构建图分类-图神经网络-11分类
日期：2021-03-24
版本：1
"""
import random
import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Data, InMemoryDataset, DataLoader
from torch_geometric.nn.glob.glob import global_max_pool
from torch_scatter import scatter_mean, scatter_max
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv, GatedGraphConv,TopKPooling, SAGPooling
import torch.optim as optim
from createDataSet import MyInMemoryDataset


torch.cuda.current_device() 
torch.cuda.empty_cache()
torch.cuda._initialized = True
#%%
class model(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_classes=12):
        """
        Arguments:
        ------
            input_dim {int} -- 输入特征的维度
            hidden_dim {int} -- 隐藏层单元数

        Keyword Arguments:
        ------
            num_classes {int} -- 分类类别数 (default: {12})
        """ 
        super(model, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes

        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.pool1 = SAGPooling(hidden_dim, 0.5)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.pool2 = SAGPooling(hidden_dim, 0.5)  
        self.conv3 = GCNConv(hidden_dim, hidden_dim)
        self.pool3 = SAGPooling(hidden_dim, 0.5)

        #self.lstm_hidden = self.init_lstm_hidden().to(device)
        self.lstm = nn.LSTM(1,1,num_layers=2,batch_first=True,bidirectional=True)    # 双向lstm
        
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, num_classes))

    def init_lstm_hidden(self):
        # 各个维度的含义是 (num_layers, minibatch_size, hidden_dim)
        return (torch.zeros(4, 10, 1),
                torch.zeros(4, 10, 1))
        

    def forward(self, data):  #
        indexs = torch.tensor(list(range(2000))*10) 
        x, edge_index, edge_attr, batch = data.x, data.edge_index, data.edge_attr, data.batch  
        #print(x.shape)
        x1 = self.conv1(x, edge_index, edge_attr)  
        x1 = F.relu(x1)                 
        pool1, edge_index1, edge_attr1, batch1, perm1, score1 = self.pool1(x1, edge_index, edge_attr, batch)
        indexs = indexs[perm1]
        global_pool1 = torch.cat([global_avg_pool(pool1, batch1),global_max_pool(pool1, batch1)],dim=1)

        x2 = self.conv2(pool1, edge_index1, edge_attr1)
        x2 = F.relu(x2)
        pool2, edge_index2 ,edge_attr2, batch2, perm2, score2 = self.pool2(x2, edge_index1, edge_attr1, batch1)
        indexs = indexs[perm2]
        global_pool2 = torch.cat([global_avg_pool(pool2, batch2),global_max_pool(pool2, batch2)],dim=1)

        x3 = self.conv3(pool2, edge_index2, edge_attr2)
        x3 = F.relu(x3)
        pool3, edge_index3 ,edge_attr3, batch3, perm3, score3 = self.pool3(x3, edge_index2, edge_attr2, batch2)
        indexs = indexs[perm3]
        global_pool3 = torch.cat([global_avg_pool(pool3, batch3),global_max_pool(pool3, batch3)],dim=1)

        readout = global_pool1 + global_pool2 + global_pool3
        lstmout,_ = self.lstm(readout.view(len(readout),200,-1))
        #print("lstmout shape:",lstmout.shape)
        lstmout = lstmout.view(10,200,-1)
        lstmout = torch.mean(lstmout,dim=2).view(10,-1)

        logits = self.mlp(lstmout)
        
        return F.log_softmax(logits,dim=1), indexs.view(10,-1)

def global_max_pool(x, batch):
    num = batch.max().item() + 1
    return scatter_max(x,batch,dim=0,dim_size=num)[0]

def global_avg_pool(x, batch):
    num = batch.max().item() + 1 
    return scatter_mean(x, batch, dim=0, dim_size=num)


dataset = MyInMemoryDataset(root='/lustre/home/acct-clsdqw/clsdqw-jiangxue/lzq/gnn_nosex/')
loader = DataLoader(dataset, batch_size=10, shuffle=True)

test_dataset = MyInMemoryDataset(root='/lustre/home/acct-clsdqw/clsdqw-jiangxue/lzq/gnn_nosex/test/')
test_loader = DataLoader(test_dataset, batch_size=10, shuffle=True)

model = model(input_dim=200, hidden_dim=100)
print(model)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)
model.to(device)
#lossf = nn.CrossEntropyLoss().to(device)
lossf = nn.NLLLoss().to(device)
optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=5e-4)
epoch_counter = 0


for epoch in range(100):
    model.train()
    #print('-------------model train---------------')
    # 定义损失函数和优化器
    #lossf = nn.CrossEntropyLoss().to(device)
    #optimizer = optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
    
    for data in loader:
        data.to(device)
        
        # 前向计算+计算损失
        out,_ = model(data) 
        loss = lossf(out, data.y)
        # 优化器初始化+反向传播+优化器迭代优化
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    model.eval()
    ind = torch.tensor([],dtype=torch.long)
    #print('-------------model eval-----------------')
    accs = []
    length = len(test_loader)
    for test_data in test_loader:
        
        test_data.to(device)
        
        out,indexs = model(test_data)
        ind = torch.cat([ind,indexs],dim=0)
        _, pred = torch.max(out,dim=1)
        correct = (pred==test_data.y).sum().item()
        acc = correct / len(data.y)
        accs.append(acc)

    acc = np.mean(accs)
    # 打印输出准确率
    print('Epoch {:03d} train_loss: {:.4f} train_acc: {:.4f}'.format(
        epoch, loss.item(), acc))
    ind = ind.numpy()
    ind_cor_mat = np.corrcoef(ind)
    ind_cor_mat_sum = np.max(np.sum(ind_cor_mat,axis=0))
    if epoch_counter == 0:
        cor_mat_sum = ind_cor_mat_sum
        index_selected = ind
    if epoch_counter > 0:
        if ind_cor_mat_sum > cor_mat_sum:
            cor_mat_sum = ind_cor_mat_sum
            index_selected = ind
            print(index_selected.shape)
    epoch_counter += 1

df = pd.DataFrame(index_selected)
df.to_csv('perms.csv')
