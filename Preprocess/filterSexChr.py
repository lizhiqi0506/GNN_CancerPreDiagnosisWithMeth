#coding:utf-8

import numpy as np
import pandas as pd

df = pd.read_csv('/lustre/home/acct-clsdqw/clsdqw-jiangxue/lzq/gnn2/data/filter_zero_meth_matrix.csv',index_col=0)
aSample_df = pd.read_csv('/lustre/home/acct-clsdqw/clsdqw-jiangxue/lzq/gnn_nosex/data/aSample.txt',sep='\t')
print(df.shape)
filter_sites = df.iloc[:,0].values
aSample_df = aSample_df[aSample_df['Composite Element REF'].isin(filter_sites)]
print(aSample_df.shape)
sites = aSample_df.iloc[:,0]
notsexChrSites = sites[(aSample_df['Chromosome'].isin(['chrX','chrY']))==False].values
#notsexChrSites = aSample_df[(aSample_df['Chromosome'].isin(['chrX','chrY']))==False]['composite element ref'].values
print(len(notsexChrSites))
df.set_index('composite element ref',inplace=True)
df = df.loc[notsexChrSites,:]
print(df.shape)
df.to_csv('./data/filter_zero_meth_matrix_nosex.csv')

