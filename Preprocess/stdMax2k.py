#coding:utf-8

import pandas as pd
import numpy as np
import os

# lzq
# select the top 20k methylation sites with max standard deviation
# 2021-01-29

def main():
    beta_matrix_df = pd.read_csv('./data/filter_zero_meth_matrix_nosex.csv')    # 读取甲基化矩阵
    print(beta_matrix_df.shape)
    meth_site_num = len(beta_matrix_df)         # 甲基化位点总数
    beta_values = beta_matrix_df.values[:,1:]

    stds = []
    for row in beta_values:
        std = np.std(row)
        stds.append(std)

    beta_matrix_df['standard deviation'] = stds     # 甲基化矩阵添加新列std
    beta_matrix_df = beta_matrix_df.sort_values(by='standard deviation',ascending=False)    # 按std降序排序
    beta_matrix_df_max2k = beta_matrix_df.head(2000).reset_index(drop=True).iloc[:,:-1]       # 提取std最大的前2k
    beta_matrix_df_max2k.to_csv('./data/meth_matrix_maxstd_2k.csv')
    """
    std_max20k = beta_matrix_df['standard deviation'].head(20000).reset_index(drop=True)
    std_max20k.to_csv('/lustre/home/acct-clsdqw/clsdqw-jiangxue/lzq/dataPreprocess/std_max20k.csv')
    """
if __name__ == '__main__':
    main()




    
