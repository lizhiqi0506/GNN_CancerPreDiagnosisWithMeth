#%%

#coding:utf-8

import numpy as np
import pandas as pd
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import seaborn as sns


#%%
pca_mat = pd.read_csv('./data/pca_top100_nosex.csv',index_col=0)
X = pca_mat.values.T    # shape=(6900*100)
#labels = [0]*438 + [1]*686 + [2]*886 + [3]*914 + [4]*309 + [5]*343 + [6]*474 + [7]*872 + [8]*468 + [9]*542 + [10]*398 + [11]*570
labels = ['Bladder']*438 + ['Brain']*686 + ['Breast']*886 + ['Bronchus']*914 + ['Cervix']*309 + \
    ['Corpus uteri']*474 + ['Kidney']*872 + ['Liver']*468 + ['Prostate']*542 + ['Stomach']*398 + ['Thyroid']*570
labels = np.array(labels)
tsne = TSNE()
#%%
X_embedded = tsne.fit_transform(X)

colors = ['blue','orange','green','red','purple','saddlebrown','deeppink','gray','gold','aqua','black']
sns.set_context('paper')
sns.set(rc={'figure.figsize':(15,10)})
palette = sns.color_palette(colors)
sns.set_style('white')

fig = sns.scatterplot(X_embedded[:,0], X_embedded[:,1], hue=labels, legend='full', palette=palette)
fig.legend(loc='upper right',ncol=2,borderaxespad=0.3,fontsize=14,markerscale=1.5)
plt.xlabel('tSNE1',fontsize=16,fontweight='bold')
plt.ylabel('tSNE2',fontsize=16,fontweight='bold')
plt.xticks([])
plt.yticks([])
plt.rcParams['figure.dpi'] = 600 #分辨率

fig_path = './data/tsne_nosex.png'
scatter_fig = fig.get_figure()
scatter_fig.savefig(fig_path, dpi = 600)

# %%
