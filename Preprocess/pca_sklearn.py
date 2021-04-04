#!/usr/bin/python
#coding:utf-8
"""
作者：lzq
功能：pca using sklearn package
日期：2021-02-26
"""
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

def main():
    meth_matrix_max2k_df = pd.read_csv('./data/meth_matrix_maxstd_2k.csv', index_col=0)
    meth_matrix_max2k_T = meth_matrix_max2k_df.values[:,1:].T
    print(meth_matrix_max2k_T.shape)

    pca = PCA(n_components=100)
    pca.fit(meth_matrix_max2k_T)
    print("explained variance ratio:",pca.explained_variance_ratio_)
    pca_vectors_top100 = pca.transform(meth_matrix_max2k_T).T
    col_name = meth_matrix_max2k_df.columns[1:]
    pca_top100_df = pd.DataFrame(pca_vectors_top100,columns=col_name)
    pca_top100_df.to_csv('./data/pca_top100.csv')

if __name__ == '__main__':
    main()


