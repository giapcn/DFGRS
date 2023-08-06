''' Created by Giap
    xay dung khung de test tinh chinh xac cuar thuat toan GRS cho separate user
    + muc dich la test cac phep toan suy dien tren Tb_IFE cho GRS
'''
import datetime as dt
import time_convertion as tc
import D_IF_GRS
import Tb_IFE
import IFE_fuzzilizer
import IFE
import pandas as pd
import numpy as np
import copy
import group_generator as gg

''' predict the individual user's preference'''
def predict(User_index, D_IF_GRS_ins, Neighbours, err):
    #initial error
    accumulative_error=0
    error_counting=0
    # get user's rating and mean
    user_rating = copy.deepcopy(D_IF_GRS_ins.getUser_by_index(User_index))  # create a shallow copy
    user_mean_rating = D_IF_GRS_ins.getBofferoni_mean(User_index)
    print("User: ", User_index)
    print("User mean rating ", user_mean_rating.show())
    # calculate neighbours' means and neighbours' simlirarities
    neighbor_means = []
    neighbor_weights = []
    for i in range(Neighbours.shape[0]):
        neighbor_mean = D_IF_GRS_ins.getBofferoni_mean(Neighbours.iloc[i, 0])
        print('Neighbour ', i, ': ', Neighbours.iloc[i, 0])
        print('Neighbour Bofferony mean')
        neighbor_mean.show()
        neighbor_means.append(neighbor_mean)
        neighbor_weight = D_IF_GRS_ins.getSimilarity(User_index, Neighbours.iloc[i, 0])
        neighbor_weights.append(neighbor_weight)

    # predict user preference on the unrated items
    items_number = len(user_rating.membership)
    print('Number of items:', items_number)
    unrateditems = 0

    for i in range(items_number):
        timepoint_prediction_value = 0  # sử dụng để xác định các giá trị do dự báo sinh ra

        # #if True: # du bao voi moi item #user_rating.isNull(i) == True:
        if user_rating.isNull(i) == False:
            timepoint_prediction_value = user_rating.getTimepoint_by_index(i)
            print(" notice ")
        else:
            timepoint_prediction_value = -1
        # foreach unknown item-rating
        unrateditems += 1
        additive = user_mean_rating
        addjust = None

        # calculate the effect of each neighbour
        for j in range(Neighbours.shape[0]):
            neighbour_rating = D_IF_GRS_ins.getUser_Item_by_index(j, i)
            if neighbour_rating is not None:
                j_mean = neighbor_means[j]
                sim = Neighbours.iloc[j, 1]  # timebase similarity
                # print("similarity ", j, " = ", sim)
                addjust = neighbour_rating.subtract(j_mean)
                # print ("adjustment before ")
                # addjust.show()
                addjust = addjust.multiple(sim)
                # print("adjustment after")
                # addjust.show()
                additive = additive.sum(addjust)
                # print(" additive ", additive)
        # calculate error before updating
        if user_rating.isNull(i) == False:
            real_rate = user_rating.getElement(i)
            if addjust is not None:
                predicted_rate = IFE.IFE(additive.membership, additive.nonmembership, additive.hesitance)
                accumulative_error += real_rate.disEuclidean(
                    predicted_rate)  # su dung khoang cach Euclidian trong huanluyen mo hinh
                error_counting += 1
            else:
                predicted_rate = IFE.IFE(additive.membership, additive.nonmembership, additive.hesitance)
                accumulative_error += real_rate.disEuclidean(
                    predicted_rate)  # su dung khoang cach Euclidian trong huanluyen mo hinh
                error_counting += 1
        # update rating value for the item i
        if addjust is not None:
            print(" update by addjust value: ", i)
            # additive.show()
            user_rating.setElement(i, additive.membership, additive.nonmembership, additive.hesitance,
                                   timepoint_prediction_value)
        else:
            print(" update by mean ", i)
            # user_mean_rating.show()
            user_rating.setElement(i, user_mean_rating.membership, user_mean_rating.nonmembership,
                                   user_mean_rating.hesitance, timepoint_prediction_value)
    # print(" Predict rating: ")
    # user_rating.show()
    # user_rating.save("temp.csv")
    print("unrated items: ", unrateditems)
    # U_rating.to_csv("u_rating_10.csv")
    err.append(accumulative_error)
    err.append(error_counting)
    return user_rating

''' predict preference of group members '''

def group_prediction(G, Groupsize, Matrix, err_metrix, KnearestNeighbour=30, TimeFrequency=86400, TimeLamda=0.001,):
    # initial error
    accumulative_error = 0
    error_counting = 0

    # get data from file
    ins = D_IF_GRS.D_IF_GRS(0)
    ins.inputdata()  # using default file name
    ins.show()
    # G number of group, Matrix: input matrix
    for i in range(G.shape[0]):
        # for each group
        G_rating = []
        neighbour_size = KnearestNeighbour  # 5% of population
        for j in range(G.shape[1]):
            user_index = G.iloc[i, j]
            user_Neighbor = ins.getNeighbour_timebase(user_index, neighbour_size, TimeFrequency, TimeLamda)
            err_ins = []
            U_rating = predict(user_index, ins, user_Neighbor, err_ins)
            G_rating.append(np.append(user_index, U_rating.getMembership()))
            G_rating.append(np.append(user_index, U_rating.getNonMembership()))
            G_rating.append(np.append(user_index, U_rating.getHesitance()))
            G_rating.append(np.append(user_index, U_rating.getTimepoint()))
            accumulative_error += err_ins[0]
            error_counting += err_ins[1]
        print(" group i")
        print(G_rating)
        df_G_rating = pd.DataFrame(data=G_rating)
        filename = 'DIFGRS_results\\group_rating\\' + str(Groupsize) + '_group_rating_' + str(i) + '.csv'
        df_G_rating.to_csv(filename)
    err_metrix.append(accumulative_error)
    err_metrix.append(error_counting)

def predict_group_rating(pref_matrix, UserList,K_nearest_neighbour, TimeFrequency, TimeLamda, Error):

    # training with random generated groups
    size = [3, 5, 10, 15, 20]
    # initial error
    accumulative_error = 0
    error_counting = 0
    #
    for i in range(len(size)):
        group_size = size[i]
        print('group size =', group_size)
        group_number = 30
        Groups = gg.groupsGenerator(UserList.tolist(), group_size, group_number)
        Groups = Groups - 1  # convert user_id to user_index
        Group_filename = 'DIFGRS_results\\group_profile_groupsize' + str(group_size) + '_number_' + str(
            group_number) + '.csv'
        G_profile = pd.DataFrame(data=Groups)
        G_profile.to_csv(Group_filename)
        print(Groups)
        Error_metrix = []
        group_prediction(G=Groups, Groupsize=group_size, Matrix=pref_matrix, err_metrix= Error_metrix, KnearestNeighbour=K_nearest_neighbour,
                         TimeFrequency=TimeFrequency, TimeLamda=TimeLamda)
        accumulative_error += Error_metrix[0]
        error_counting += Error_metrix [1]
    Error.append(accumulative_error)
    Error.append(error_counting)


if __name__ == '__main__':

    #############
    # get transform data of MovieLens 1M
    pref_matrix = pd.read_csv("movielens.csv")

    # generate all testing groups
    UserList = pref_matrix['userId']
    print(UserList)
    ###########################
    K = 55 # 10% of the population
    TimeFrequency = 86400* 5
    TimeLamda = 0.0001
    err = []
    predict_group_rating(pref_matrix, UserList, K, TimeFrequency, TimeLamda, err)
    # GRS_check
    # temp(K, TimeFrequency, TimeLamda, err)
    print("ket qua trung ",err[0], " error = ", err[0] / err[1])

