''' Created by Giap'''
import numpy as np
import pandas as pd
import Tb_IFE
import IFE
import IFE_fuzzilizer
import time_convertion
#data structured of DF_GRS

class D_IF_GRS:
    def __init__(self, initialUser=0):
        self.data = [] #np.array(dtype = np.dtype(Tb_IFE.Tb_IFvector))
        self.users = initialUser

    def inputdata(self, Rating_file='movielens.csv', Timestamp_file='time_pre_movielens.csv'):
        # read two file
        # get transform data of MovieLens 1M
        rate_matrix = pd.read_csv(Rating_file)
        timestamp_matrix = pd.read_csv(Timestamp_file)
        # print(rate_matrix)

        # generate all user preference data based on Tb_IFE
        userList = rate_matrix['userId']
        print(userList)
        self.users= len(userList)

        item_number= rate_matrix.shape[1]
        for i in range(self.users): #userList.size):
            newnode = Tb_IFE.Tb_IFvector(initialEles=item_number-1)
            memberships = np.zeros(item_number - 1, dtype=float)
            non_memberships = np.zeros(item_number - 1, dtype=float)
            hesitances = np.zeros(item_number - 1, dtype=float)
            timepoints = np.zeros(item_number - 1, dtype=float )

            for j in range(1, item_number): #start from column index 1
                rating = rate_matrix.iloc[i, j]
                # print(rating)
                #notice: put item j into position [j-1] in the list
                if pd.isnull(rate_matrix.iloc[i, j]):
                    memberships[j-1] = np.NaN
                    non_memberships[j-1] = np.NaN
                    hesitances[j-1] = np.NaN
                    timepoints[j-1] = np.NaN
                else:
                    fuzzilizer = IFE_fuzzilizer.IFE_fuzzilier()
                    memberships[j-1] = fuzzilizer.mu(rating)
                    non_memberships[j-1] = fuzzilizer.nu(rating)
                    hesitances[j-1] = fuzzilizer.pi(rating)
                    timepoints[j-1] = timestamp_matrix.iloc[i,j]
            newnode.setEles(memberships, non_memberships, hesitances, timepoints)
            self.data.append(newnode)

    def append_data(self,Node):
        self.data.append(Node)
    def show(self):
        for i in range(len(self.data)):
            self.data[i].show()

    def getUser_by_index(self,index):
        return self.data[index]

    #get user preference on an item
    def getUser_Item_by_index(self,User_index, Item_index):
        #return None incase missing value
        result= self.data[User_index].getElement(Item_index)
        return result

    # def distanc(self, position1, position2):
    #     length = self.data.count()
    #     if position1<length and position2 <length:
    #         dis = self.data[position1]
    #         print(dis)

    #get K neighbours of user u
    def getNeighbour_timebase(self,u, K, TimeFrequency=86400, TimeLamda=0.001):
        Neigbours = []
        count = 0
        # find K neighbour of u
        for i in range(len(self.data)):
            if i != u:
                # print(buff)
                sim = self.data[u].similarity_with_TE(self.data[i],time_Frequency=TimeFrequency, time_Lamda= TimeLamda)
                # print('sim =',sim)
                if count < K:
                    Neigbours.append([i, sim])
                    count = count + 1
                else:
                    # thay the phan tu nho nhat trong danh luc neighbour
                    maxpos = 0
                    for j in range(K):
                        if Neigbours[maxpos][1] < Neigbours[j][1]:
                            maxpos = j;
                    # replace
                    if Neigbours[maxpos][1] < sim:
                        # print(' exchange ',Neigbours[maxpos][1], sim )
                        Neigbours[maxpos][0] = i
                        Neigbours[maxpos][1] = sim
        k_neighbours = pd.DataFrame(data=Neigbours)
        return k_neighbours

    #get Bofferroni mean of specific element
    def getBofferoni_mean(self,index):
        #Bofferoni mean parameters
        p=1
        q=2
        Boff_mean =self.data[index].Bonfferroni_mean(p,q)
        return Boff_mean

    # get IAFW mean of specific element
    def getIAFW_mean(self, index):
        # Bofferoni mean parameters
        iafw_mean = self.data[index].IAFW_mean()
        return iafw_mean

    # calculate similarity between users
    def getSimilarity(self, User_index_01, User_index_02):
        #using cosine similarity
        cos_sim = self.data[User_index_01].similarity(self.data[User_index_02],sim_type="cosine")
        return cos_sim

    #
    def save_csv(self, Filename):
        # concatinate the data of all elements, and each element take place of 4 row
        G_rating = []
        for i in range(len(self.data)):
            U_rating = self.data[i]
            G_rating.append(U_rating.getMembership())
            G_rating.append(U_rating.getNonMembership())
            G_rating.append(U_rating.getHesitance())
            G_rating.append(U_rating.getTimepoint())
        print(" Saving to file thi data")
        print(G_rating)
        df_G_rating = pd.DataFrame(data=G_rating)
        df_G_rating.to_csv(Filename)

if __name__ == '__main__':
    print(" Dynamic Fuzzy GRS system ")
    ins =D_IF_GRS(0)
    ins.inputdata("none","none")
    ins.show()
    print(' element 1: ')
    print(ins.getindex(0))
    print("similarity")
    print(ins.getindex(0).similarity(ins.getindex(1)))
    print ("time-base similarity ")
    print(ins.getindex(0).similarity_with_TE(ins.getindex(1)))
    kneig=ins.getNeighbour_timebase(1,4)
    print(kneig)

    # x= Tb_IFE.Tb_IFvector(initialEles=3)
    # Membership = np.array([1.5,2,3], dtype=float)
    # Non_membership = np.array([1,2,3], dtype=float)
    # Hesitance = np.array([1,2.5,3], dtype=float)
    # Timepoint = np.array([1.5,2,3.5], dtype=int)
    # x.setEles(Membership,Non_membership,Hesitance,Timepoint)
    # x.show()
    # y = IFE.IFE(0.8,0.1,0.1)
    # y.show()
    # z = IFE.IFE(0.5, 0.2, 0.3)
    # z.show()
    # print("Modified Hamming distance <y,z> =", y.disHammingModify(z))
    # w=y.product(z)
    # w.show()
    # print(" Danh much cac user ")
    # ins =D_IF_GRS(0)
    # ins.append_data(x)
    # ins.show()