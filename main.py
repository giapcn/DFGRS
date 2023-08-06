from random import sample
from datetime import datetime
import pandas as pd
import numpy as np

import distance as my_dis
import IFE_fuzzilizer as IFEF

import csv
#
# xgboost as xgb
# from surprise import Reader, Dataset
# from surprise import BaselineOnly
# from surprise import KNNBaseline
# from surprise import SVD
# from surprise import SVDpp
# from surprise.model_selection import GridSearchCVdef
#
#
# def load_data():
#     netflix_csv_file = open("netflix_rating.csv", mode="w")
#     rating_files = ['combined_data_1.txt']
#     for file in rating_files:
#         with open(file) as f:
#             for line in f:
#                 line = line.strip()
#                 if line.endswith(":"):
#                     movie_id = line.replace(":", "")
#                 else:
#                     row_data = []
#                     row_data = [item for item in line.split(",")]
#                     row_data.insert(0, movie_id)
#                     netflix_csv_file.write(",".join(row_data))
#                     netflix_csv_file.write('\n')
#
#     netflix_csv_file.close()
# Press the green button in the gutter to run the script.
import datetime as dt
import time_convertion as tc
if __name__ == '__main__':
    temp= tc.time_convertion()
    temp.set_root(int(dt.datetime.now().timestamp())-86400100)
    #temp.set_frequency(50)
    temp.set_lamda(10)
    print("time plot: ", temp.time_transpose(int(dt.datetime.now().timestamp())))

    temp= IFEF.IFE_fuzzilier()
    for i in range(1,11):
        print(i,": ",temp.mu(i),"; ",temp.nu(i),"; ", temp.pi(i))
    # Skip data
    # df = pd.read_csv('./data/combined_data_1.txt', header=None, names=['Cust_Id', 'Rating'], usecols=[0, 1])
    #
    # df['Rating'] = df['Rating'].astype(float)
    #
    # print('Dataset 1 shape: {}'.format(df.shape))
    # print('-Dataset examples-')
    # print(df.iloc[::5000000, :])
    #
    # p = df.groupby('Rating')['Rating'].agg(['count'])
    #
    # # get movie count
    # movie_count = df.isnull().sum()[1]
    #
    # # get customer count
    # cust_count = df['Cust_Id'].nunique() - movie_count
    #
    # # get rating count
    # rating_count = df['Cust_Id'].count() - movie_count
    #
    # ax = p.plot(kind='barh', legend=False, figsize=(15, 10))
    # plt.title(
    #     'Total pool: {:,} Movies, {:,} customers, {:,} ratings given'.format(movie_count, cust_count, rating_count),
    #     fontsize=20)
    # plt.axis('off')
    #
    # for i in range(1, 6):
    #     ax.text(p.iloc[i - 1][0] / 4, i - 1, 'Rating {}: {:.0f}%'.format(i, p.iloc[i - 1][0] * 100 / p.sum()[0]),
    #             color='white', weight='bold')

    #plt.imshow()

    df = pd.read_csv('./data/ratings.csv')
    # df['rating'] = df['rating'].astype(float)
    # #print(df)
    # # drop duplicate
    # # newdf = df.drop_duplicates(subset="userId")
    # # print(newdf)
    #
    # #print(df[df.duplicated('userId')])
    #
    # mean_rating = df['rating'].mean()
    # print(" mean reating =",mean_rating)
    pref_matrix = df[['userId', 'movieId', 'rating']].pivot(index='userId', columns='movieId', values='rating')
    print(pref_matrix.dtypes)
    # pref_matrix.to_csv("movielens.csv")
    '''
    pref_matrix = pref_matrix - mean_rating  # adjust by overall mean

    item_mean_rating = pref_matrix.mean(axis=0)
    pref_matrix = pref_matrix - item_mean_rating  # adjust by item mean

    user_mean_rating = pref_matrix.mean(axis=1)
    pref_matrix = pref_matrix - user_mean_rating
    print(pref_matrix)
    #tao danh muc nhom bang random sampling mechanism 
    List = [1,2,3,4,5,6,7,8,9,10,11]
    for i in range(3):
        print(sample(List,3))
    '''
    # df = pd.read_csv('./data/movies.csv')
    # movies = df.join(df.genres.str.get_dummies("|"))
    # sim = cosine_similarity(movies.iloc[:, 3:]) #su dung cac tham so tu cot thu 4
    # toystory_top5 = np.argsort(sim[0])[-5:][::-1]
    # print(movies.iloc[toystory_top5])
    # (movies.iloc[toystory_top5]).to_csv("giap.csv")

    # df = pd.read_csv('./data/book-crossing/Ratings.csv')
    # df=df.drop_duplicates(subset=['book_id', 'user_id'], keep='last')
    # print(df)
    # pref_matrix = df.pivot(index='user_id', columns='book_id', values='rating')
    # print(pref_matrix)
    # pref_matrix.to_csv("book-crossing.csv")

    # df = pd.read_csv('./data/book-crossing/Ratings.csv',sep=';')
    #print(df.head())
    # df = df.iloc[0:10000,:]
    #print(df)
    # pref_matrix = df[['User-ID', 'ISBN', 'Rating']].pivot(index='User-ID', columns='ISBN', values='Rating')
    # print(pref_matrix)
    #pref_matrix.to_csv("book-crossing.csv")
    buff= pref_matrix.iloc[[0,1],:]
    print(buff)
    print(" Kieu du lieu trong bang ")
    print(buff.dtypes)
    # buff.iloc[0, 8] = 15
    # buff.iloc[0,[2,7]]=5
    # buff.iloc[1,[7,8]] = 3
    # buff.iloc[1, 2]=1
    # buff=buff.transpose();
    # print(buff)
    # buff= buff.dropna()
    # buff=buff.T
    # # buff=buff.dropna(how='all', axis=1, inplace=True)
    # print(buff)
    print(my_dis.vector_cosine_sim(buff))

    buff1 = pd.DataFrame(columns=['A','B','C'])
    buff1=buff1.append({'A':2,'B':4,'C':80},ignore_index=True)
    buff1=buff1.append({'A': 3, 'B': 9, 'C': 4},ignore_index=True)
    buff1=buff1.append({'A': 5, 'B': 30, 'C': 6},ignore_index=True)
    buff1=buff1.append({'A': 1, 'B': 4, 'C': 65},ignore_index=True)
    buff1 = buff1.append({'A': 5, 'B': 30, 'C': 6}, ignore_index=True)
    buff1 = buff1.append({'A': 1, 'B': 4, 'C': 65}, ignore_index=True)
    buff1 = buff1.append({'A': 3, 'B': 10, 'C': 4}, ignore_index=True)
    print(buff1)
    # mahanobis = my_dis.mahalanobis(x=buff1,data=buff1[['A','B','C']])
    newdata = pref_matrix.iloc[:10, :]
    #newdata=newdata.T
    newdata=newdata.fillna(0.0) #newdata.dropna()
    print(newdata)
    # newdata=newdata.T
    # print("newdata",newdata)
    mahanobis = my_dis.mahalanobis_dis(newdata)
    # mahanobis=my_dis.temp_ma(buff1)
    print("khoang cach ")
    print(mahanobis)
    print("means",mahanobis.mean())
    if(np.isnan(mahanobis.mean())):
        print(" haha")



