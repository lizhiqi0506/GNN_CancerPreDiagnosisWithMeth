#coding:utf-8

from numpy.core.numeric import NaN
import pandas as pd
import numpy as np
import os

# lzq
# get the NaN lines' Index
# 2021-03-01

dir = '/lustre/home/acct-clsdqw/clsdqw-jiangxue/TCGA/methylation/'

basefiledir = '/lustre/home/acct-clsdqw/clsdqw-jiangxue/lzq/dataPreprocess/file_alignment.csv'
basefile = pd.read_csv(basefiledir,index_col=0)
basefile = basefile.sort_values(by='Site Code')     # 按primary site排序
basefile_values = basefile.values
filelist = basefile_values[:,0]    #按primary site排序后所有文件的文件名

isnan = np.array([True]*485577)
for i in range(len(filelist)):
    print(i)
    subject_code = filelist[i]
    path = dir + subject_code
    df = pd.read_csv(path,sep='\t')
    beta_values = df['Beta_value']
    isnan = isnan & np.isnan(beta_values).values

nanLines = np.where(isnan)[0]
print("Num of NaN Lines:",len(nanLines))

pd.DataFrame(nanLines).to_csv("/lustre/home/acct-clsdqw/clsdqw-jiangxue/lzq/dataPreprocess/NaNLineIndex.csv")
