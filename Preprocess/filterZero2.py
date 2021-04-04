#coding:utf-8

import numpy as np
import pandas as pd

beta_value_matrix_df = pd.read_csv('/lustre/home/acct-clsdqw/clsdqw-jiangxue/lzq/dataPreprocess/beta_value_matrix.csv',index_col=0)
nalines = pd.read_csv('/lustre/home/acct-clsdqw/clsdqw-jiangxue/lzq/dataPreprocess/NaNLineIndex.csv',index_col=0).values
nalines = nalines[:,0]
filter_beta_value_matrix_df = beta_value_matrix_df.drop(beta_value_matrix_df.index[nalines])
filter_beta_value_matrix_df = filter_beta_value_matrix_df.reset_index(drop=True)
filter_beta_value_matrix_df.to_csv('/lustre/home/acct-clsdqw/clsdqw-jiangxue/lzq/dataPreprocess/filter_meth_matrix.csv')
