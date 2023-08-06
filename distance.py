''' Giap
    using to calculate some distance type using in GRS
'''


import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from scipy.spatial.distance import mahalanobis

def vector_cosine_sim( X):
    # consine similar with out empty cells
    buff = X.transpose();
    buff = buff.dropna()
    buff = buff.T
    # print("lean data",buff)
    buff=buff.iloc[:,1:] #remove user id
    if buff.empty == False:
        cos_sim = cosine_similarity(buff)
        # print('distance',cos_sim)
        return cos_sim[0,1]
    else:
        return 0 # with empty vectors return zero value

#create function to calculate Mahalanobis distance
def mahalanobis_my(x=None, data=None, cov=None):
    x_mu = x - np.mean(data)
    print(data.T)
    if not cov:
        cov = data.T.cov()
    print(cov)
    inv_covmat = np.linalg.inv(cov)
    left = np.dot(x_mu, inv_covmat)
    mahal = np.dot(left, x_mu.T)
    return mahal.diagonal()

def mahalanobis_dis(df):
    '''gives series mahalanobis distance of object to center of group'''
    means = df.mean()
    # print("means",means)
    # inv_cov = np.linalg.inv(df.astype(float).cov())
    # print(inv_cov)
    try:
        inv_cov = np.linalg.inv(df.astype(float).cov())
    except np.linalg.LinAlgError:
        return pd.Series([np.NAN] * len(df.index), df.index,
                         name='Mahalanobis')
    # print(inv_cov)
    dists = []
    for i, sample in df.iterrows():
        # print("sample",sample)
        ma_dis = mahalanobis(sample, means, inv_cov)
        # print('mah dis',ma_dis)
        dists.append(ma_dis)
    return pd.Series(dists, df.index, name='Mahalanobis')
