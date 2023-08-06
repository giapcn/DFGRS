''' created by Giap
    Testing on MovieLens 1M
'''

import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity
import distance as my_dis
import csv

if __name__ == '__main__':
    '''convert the orginal data into the matrix of user-item rating'''
    # pre-process data
    df = pd.read_csv('./data/ratings.csv')
    df['rating'] = df['rating'].astype(float)
    df['timestamp'] = df['timestamp'].astype(int)

    # mean_rating = df['rating'].mean()
    # print(" mean reating =",mean_rating)

    # pref_matrix = df[['userId', 'movieId', 'rating']].pivot(index='userId', columns='movieId', values='rating')
    # print(pref_matrix)
    # pref_matrix.to_csv("movielens.csv")

    timestamp_matrix = df[['userId', 'movieId', 'timestamp']].pivot(index='userId', columns='movieId', values='timestamp')
    timestamp_matrix= (timestamp_matrix/86400).fillna(0).astype(int)
    print(timestamp_matrix)
    timestamp_matrix.to_csv("time_pre_movielens.csv")