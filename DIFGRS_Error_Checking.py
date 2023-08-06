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

def GRS_check():
    print(" testing the accuracy of the GRS based on IFE ")
    # read the rating matrix
    # get data from  ;  convert to IFE matrix
    ins = D_IF_GRS.D_IF_GRS(0)
    ins.inputdata()  # using default file name
    ins.show()

    # reading the predict matrix and calculate the accuracy
    defuzzilier = IFE_fuzzilizer.IFE_defuzzilier()

    sum_error= 0;
    count = 0;

    size = [3, 5, 10, 15, 20]
    for i in range(len(size)):
        group_size = size[i]
        group_number = 1
        for j in range(group_number):
            filename = 'DIFGRS_results\\group_rating\\' + str(group_size) + '_group_rating_' + str(j) + '.csv'
            print(" Filename: ", filename)
            # DIFGRS_test(filename)
            users_rate = []
            users_index = []
            df = pd.read_csv(filename)
            numberofmembers = int(df.shape[0] / 4)  # thông tin về một user được lưu trên 4 dòng (timed-IF)
            for i in range(numberofmembers):
                user_index = df.iloc[4 * i + 0, 1]
                users_index.append(int(user_index))
                membership = df.iloc[4 * i + 0, 2:]
                nonmembership = df.iloc[4 * i + 1, 2:]
                hesitance = df.iloc[4 * i + 2, 2:]
                timestamp = df.iloc[4 * i + 3, 2:]
                # add to the TB_IFVector instance
                print(" valid item ", len(membership))
                user_ins = Tb_IFE.Tb_IFvector(len(membership))
                user_ins.setEles(membership, nonmembership, hesitance, timestamp)
                users_rate.append(user_ins)
            print("user index ")
            print(users_index)
            print("user rating")
            for i in range(len(users_rate)):
                #users_rate[i].show()
                for j in range(users_rate[i].eles_num):
                    obj = users_rate[i].getElement(j)
                    UI_i_j_deffuzzification = defuzzilier.getScriptation(obj)
                    if(users_rate[i].getTimepoint_by_index(j)!= -1):
                        print(" rating score ",j," = ",UI_i_j_deffuzzification)
                        real_rate = ins.getUser_Item_by_index(users_index[i],j)
                        print(" difference user ", users_index[i], " on item ", j, " = ",real_rate.disHamming(obj))
                        print("real_rate")
                        real_rate.show()
                        print("predicted rate")
                        obj.show()
                        sum_error += real_rate.disHamming(obj)
                        count += 1

    print(" ket qua trung ", count, " error = ", sum_error/count)


if __name__ == '__main__':
    # predict_group_rating()
    GRS_check()
