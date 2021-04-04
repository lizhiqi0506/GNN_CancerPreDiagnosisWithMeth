### Graph neural network learning model for precision multi-tumour early diagnostics with DNA methylation data

---

#### Demo

##### Data Preprocess

| Function                                      | Script                            | Result File                       |
| --------------------------------------------- | --------------------------------- | --------------------------------- |
| 将样本文件与临床信息合并                      | Preprocess/clinical/merge_file.py | merged_file.csv                   |
| 由merged_file.csv统计各类样本信息             | Preprocess/clinical/compare.py    | sample_info.csv                   |
| 构建全样本甲基化矩阵                          | Preprocess/methMatrix.py          | meth_matrix.csv                   |
| 统计β值在所有样本中都为0的位点索引            | Preprocess/NaNLines.py            | NaNLinesIndex.csv                 |
| 删除meth_matrix.csv中β=0的行                  | Preprocess/filterZero2.py         | filter_meth_matrix.csv            |
| 删除甲基化矩阵中在性染色体上的位点            | Preprocess/filterSexChr.py        | filter_zero_meth_matrix_nosex.csv |
| 提取β值在所有样本中标准差最大的2k个甲基化位点 | Preprocess/stdMax2k.py            | meth_matrix_maxstd_2k.csv         |
| PCA提取100个主成分                            | Preprocess/pca_sklearn.py         | pca_top100.csv                    |
| 由pca_top100.csv进行TSNE可视化分析            | Preprocess/tsne.py                | tsne.png                          |

##### GNN

***不同类组织网络分类***

| Function                                             | Script                      | Result File                                                  |
| ---------------------------------------------------- | --------------------------- | ------------------------------------------------------------ |
| 构建图数据集                                         | diffTissue/createDataSet.py | diffTissue/processed/dataset_diffTissue.pt、diffTissue/test/processed/dataset_diffTissue.pt |
| 训练图神经网络、获得特征位点在maxstd2k的位点中的索引 | diffTissue/gnn.py           | gnn.out、perms.csv                                           |
| 用TCGA和GEO的注释文件为所选位点进行注释              | jupyter notebook中完成      | meth_sites_selected_200.csv、meth_selected_200_info.csv      |

***同类组织样本的case-normal分类***

| Function                                             | Script                                  | Result File                                                  |
| ---------------------------------------------------- | --------------------------------------- | ------------------------------------------------------------ |
| 构建图数据集                                         | case_normal/createDataSet_casenormal.py | case_normal/processed/[primary_site].pt、case_normal/test/processed/[primary_site].pt |
| 训练图神经网络、获得特征位点在maxstd2k的位点中的索引 | case_normal/gnn_casenormal.py           | gnn_casenormal.out、case_normal/prems/perms-[primary-site].csv |

***不同类组织样本分类（节点分类）***

| Function                                                     | Script                                      | Result File                                           |
| ------------------------------------------------------------ | ------------------------------------------- | ----------------------------------------------------- |
| 由perms.csv和meth_matrix_maxstd_2k.csv构建所选200个位点的甲基化矩阵 | jupyter notebook 中完成                     | meth_matrix_selected_200.csv                          |
| 构建图数据集                                                 | node_classify/createDataset_nodeclassify.py | node_classify/processed/<br />dataset_nodeclassify.pt |
| 训练图神经网络                                               | node_classify/gnn_nodeclassify.py           | node_classify/gnn_nodeclassify.out                    |

##### 注：部分数据文件与数据集由于文件过大未上传

* 不同类癌症组织网络分类

  * 每类构建图：训练集30（总330），测试集10（总110）
  * 由R-package WGCNA得到各类样本软阈值power
  * 各位点相关系数进行power次方后与边阈值0.01比较，决定两位点之间是否存在边
  * power次方的相关系数作为边的特征值
  * 图神经网络模型如下

  ![gnn_model](gnn_model.png)

* 同类样本的case-normal分类

  * case和normal分别构建图样本数50[训练集]，20[测试集] (共100/40)
  * 各位点相关系数平方后与边阈值0.3比较，决定两位点之间是否存在边
  * 相关系数平方作为边的特征值
  * 网络模型：lstm层数变为1，全连接层变为2

* 不同类组织的样本分类

  * 构建30个图

  * 每个图中，一个节点表示一个样本

  * 每个节点具有相应的标签[0~10]，特征向量为所选200个节点的beta值

  * 设定边阈值为0.8，若两个节点特征向量的相关系数$cor$大于0.8，则两节点间存在边

  * 边的特征值为 $(cor-0.8)/(1-0.8)$

  * 构建每个图的样本为：从各类组织的样本中随机抽取50%

  * 网络模型：未使用SAGPooling，使用了3层图卷积和3个全连接层

    ```python
    class GNN(nn.Module):
        def __init__(self):  # 不用传数据参数
            super(GNN, self).__init__()
            self.conv1 = GCNConv(input_dim, hidden_dim)
            self.conv2 = GCNConv(hidden_dim, hidden_dim)  
            self.conv3 = GCNConv(hidden_dim, hidden_dim) 
            self.mlp = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.ReLU(),
                nn.Linear(hidden_dim // 2, hidden_dim // 2),
                nn.ReLU(),
                nn.Linear(hidden_dim // 2, num_classes))
            #self.conv5 = GATConv(256,dataset.num_classes,dropout=0.6)
    
        def forward(self, data):  #
            
            x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr  
            x = self.conv1(x, edge_index, edge_attr)  
            x = F.relu(x)                 
            x = F.dropout(x, training=self.training)  
    
            x = self.conv2(x, edge_index, edge_attr)
            x = F.relu(x)
            x = F.dropout(x, training=self.training)
    
            x = self.conv3(x, edge_index, edge_attr)
            x = F.relu(x)
            x = F.dropout(x, training=self.training)
    
            x = self.mlp(x)
            x = F.log_softmax(x, dim=1)  
            return x
    ```

    

  
