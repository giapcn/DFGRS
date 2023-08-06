''' create by Giap
    Testing on MovieLens 1M
'''

import pandas as pd
import numpy as np
import distance as my_dis
import IFE
import Tb_IFE
import IFE_fuzzilizer
import csv

def get_metrics(UserRating, DefuzzilizeUserRating,selected_items_index, n):
    ''' get all metrics '''
    metrics=[]

    #additve group satisfication
    additive_satis=0
    for i in range(n):
        # for each item items_index[i]
        item_add_satis=0
        # print('users_rate', DefuzzilizeUserRating.iloc[:, selected_items_index[i]])
        # print(' users rate mean ',DefuzzilizeUserRating.iloc[:, selected_items_index[i]].mean())
        item_add_satis = DefuzzilizeUserRating.iloc[:, selected_items_index[i]+1].mean()
        additive_satis=additive_satis+item_add_satis
    additive_satis=additive_satis/n
    metrics.append(additive_satis)

    # # error metric
    # error_metric = 0
    # for i in range(n):
    #     # for each item items_index[i]
    #     item_error = 0
    #     count = 0
    #     for j in range(DefuzzilizeUserRating.shape[0]):
    #         user_index = DefuzzilizeUserRating.iloc[j, 0]
    #         print("item indext ", selected_items_index[i])
    #         if pd.isnull(Matrix.iloc[user_index, selected_items_index[i]]) is not True:
    #             # co rating thuc
    #             item_error = item_error + abs(
    #                 DefuzzilizeUserRating.iloc[j, selected_items_index[i]] - Matrix.iloc[user_index, selected_items_index[i]])
    #             count = count + 1
    #     if count > 0:
    #         item_error = item_error / count
    #     print("item error ", item_error, "from ", count, " items")
    #     error_metric = error_metric + item_error
    # error_metric = error_metric / n
    # print(" error metric ", error_metric)
    # metrics.append(error_metric)

    ''' fairness metrix '''
    threshold = 4.5  #threshold to presents that an user likes a recommendation
    fuzzilier = IFE_fuzzilizer.IFE_fuzzilier()
    IF_threshold= IFE.IFE(fuzzilier.mu(threshold),fuzzilier.nu(threshold),fuzzilier.pi(threshold))
    fairness_1_metric = 0
    for i in range(n):
        # for each item items_index[i]
        print('i=',i)
        item_fairness = 0
        for j in range(len(UserRating)):
            print('j = ',j)
            if UserRating[j].getElement(selected_items_index[i]).order(IF_threshold) >=0:
                item_fairness = item_fairness + 1
        item_fairness = item_fairness / len(UserRating)
        print("item fairness_1 = ", item_fairness)
        fairness_1_metric = fairness_1_metric + item_fairness
    fairness_1_metric = fairness_1_metric / n
    print(" fairness_1 metric ", fairness_1_metric)
    metrics.append(fairness_1_metric)

    '''fairness variance'''
    fairness_2_metric = 0
    for i in range(n):
        # for each item items_index[i]
        print('i=', i)
        item_fairness = 0
        number_user = len(UserRating)
        for j in range(number_user):
            print('j = ', j)
            user_pre_ji = UserRating[j].getElement(selected_items_index[i])
            for k in range(number_user):
                user_pre_ki = UserRating[k].getElement(selected_items_index[i])
                item_fairness = item_fairness + user_pre_ji.disEuclidean(user_pre_ki)
        item_fairness = item_fairness / (number_user*number_user)
        fairness_2_metric = fairness_2_metric + item_fairness
    fairness_2_metric = fairness_2_metric / n
    print(" fairness_2 metric ", fairness_2_metric)
    metrics.append(fairness_2_metric)
    # return metrics

    '''Adjusted fairness'''
    defuzzilier = IFE_fuzzilizer.IFE_defuzzilier()
    fairness_3_metric = 0
    for i in range(n):
        # for each item items_index[i]
        print('i=', i)
        item_fairness = 0
        number_user = len(UserRating)
        sum_score=0
        for j in range(number_user):
            print('j = ', j)
            user_pre_ji = UserRating[j].getElement(selected_items_index[i])
            sum_score = sum_score + defuzzilier.getAdjustStandardScriptation(user_pre_ji)
            for k in range(number_user):
                user_pre_ki = UserRating[k].getElement(selected_items_index[i])
                print('dis ',defuzzilier.getAdjustStandardScriptation(user_pre_ji),' -', defuzzilier.getAdjustStandardScriptation(user_pre_ki) )
                item_fairness = item_fairness + abs(defuzzilier.getAdjustStandardScriptation(user_pre_ji) - defuzzilier.getAdjustStandardScriptation(user_pre_ki))
        item_fairness = item_fairness / (sum_score * 2*number_user)
        fairness_3_metric = fairness_3_metric + item_fairness
    fairness_3_metric = fairness_3_metric / n
    print(" Adjusted fairness metric ", fairness_3_metric)
    metrics.append(fairness_3_metric)

    return metrics

#select top-N candidate for the each user and merge into an entire candidate list
def get_individual_recommendation(Users_rating, N):
   # user's rating presented by an Tb_IFE
   print(" number of candidate ", N)
   candidate = []
   number_users = len(Users_rating)
   print(" number of users ", number_users)
   number_items = Users_rating[0].getNumber_elements()
   print(" Number of items ", number_items)
   # for i in range(number_users):
   #    filename = "individual"+str(i)+".csv"
   #    Users_rating[i].save(filename)

   for i in range(number_users):
      print("user ", i)
      ind_top = []
      for j in range(number_items):
         if j < N:
            ind_top.append(j)
         else:
            min = 0
            for k in range(len(ind_top)):
               if Users_rating[i].getElement(ind_top[min]).order(Users_rating[i].getElement(ind_top[k])) == 1:
                  min = k
                  # print("min position ", min)

            minvalue = Users_rating[i].getElement(ind_top[min]).score()
            currentvalue = Users_rating[i].getElement(j).score()
            if Users_rating[i].getElement(j).order(Users_rating[i].getElement(ind_top[min])) == 1:
               ind_top[min] = j  # position of item
               # print("update ", j)
      # temp = pd.DataFrame(data=ind_top)
      # filename = "temp "+ str(i)+".csv"
      # temp.to_csv(filename)
      print(" individuals ", ind_top)

      # adding to the entire list of candidates
      for j in range(len(ind_top)):
         exist = False
         for k in range(len(candidate)):
            if candidate[k] == ind_top[j]:
               exist = True
               break
         if exist == False:
            candidate.append(ind_top[j])
   return candidate

# additive utility strategy using fuzzy aggregation operation
def IFS_additive_consensus(RootFile,Users_index, Users_rating, Candidates, TopN, OrgUserRating, temp_user_ma):

    metrics_top_n = []
    for k in range(len(TopN)):
        n = TopN[k]
        print('N = ', n)
        items_index = []
        group_preference = []
        # for i in range(1,UserRating.shape[1]): # bo cot dau tien la chi so vi tri cua users
        for i in range(len(Candidates)):
            group_item_pre = Users_rating[0].getElement(Candidates[i])
            # group_item_pre is added by the next users' preference using sum function
            for j in range(1, len(Users_rating)):
                group_item_pre = group_item_pre.sum(Users_rating[j].getElement(Candidates[i]))
            group_preference.append(group_item_pre)
        # print(group_preference)
        # print(" user preference ",group_preference[0][0])
        for i in range(len(group_preference)):
            if i < n:
                # items_index.append(i+1) # vi tri phan tu i trong UserRating la i+1
                items_index.append(i)
            else:
                min = 0
                for j in range(n):
                    if group_preference[items_index[min]].order(group_preference[items_index[j]]) == 1:
                        min = j
                if group_preference[items_index[min]].order(group_preference[i]) == -1:
                    items_index[min] = i
        print(' khuyen nghi Additive average strategy')
        print(items_index)
        top_n_items = []
        for i in range(len(items_index)):
            top_n_items.append(Candidates[items_index[i]])
        print(' top n ', top_n_items)

        metrics = get_metrics( Users_rating, temp_user_ma, top_n_items, n)
        print('performance ', metrics)
        metrics_top_n.append(metrics)

    # metrics for all elements in vector top_n
    metrics_top_n = pd.DataFrame(data=metrics_top_n)

    filename = RootFile + '_IF_additive_metrics.csv'
    metrics_top_n.to_csv(filename)
    return metrics_top_n

# additive utility strategy using fuzzy aggregation operation
def IFS_multiple_consensus(RootFile,Users_index, Users_rating, Candidates, TopN, OrgUserRating, temp_user_ma):

    metrics_top_n = []
    for k in range(len(TopN)):
        n = TopN[k]
        print('N = ', n)
        items_index = []
        group_preference = []
        # for i in range(1,UserRating.shape[1]): # bo cot dau tien la chi so vi tri cua users
        for i in range(len(Candidates)):
            group_item_pre = Users_rating[0].getElement(Candidates[i])
            #group_item_pre is added by the next users' preference using product function
            for j in range(1, len(Users_rating)):
                group_item_pre = group_item_pre.product(Users_rating[j].getElement(Candidates[i]))
            group_preference.append(group_item_pre)
        # print(group_preference)
        # print(" user preference ",group_preference[0][0])
        for i in range(len(group_preference)):
         if i < n:
            # items_index.append(i+1) # vi tri phan tu i trong UserRating la i+1
            items_index.append(i)
         else:
            min = 0
            for j in range(n):
               if group_preference[items_index[min]].order(group_preference[items_index[j]]) == 1:
                  min = j
            if group_preference[items_index[min]].order(group_preference[i]) == -1:
               items_index[min] = i
        print(' khuyen nghi from Multiple strategy ')
        print(items_index)
        top_n_items = []
        for i in range(len(items_index)):
            top_n_items.append(Candidates[items_index[i]])
        print(' top n ', top_n_items)

        metrics = get_metrics(Users_rating, temp_user_ma, top_n_items, n)
        print('performance ', metrics)
        metrics_top_n.append(metrics)

    # metrics for all elements in vector top_n
    metrics_top_n = pd.DataFrame(data=metrics_top_n)

    filename = RootFile + '_IF_multiple_metrics.csv'
    metrics_top_n.to_csv(filename)
    return metrics_top_n

#Consensus using Choquet operation on intuitionistic fuzzy set
def IFS_Choquet_aggregation_consensus (RootFile,Users_index, Users_rating, Candidates, TopN, OrgUserRating, temp_user_ma):

    metrics_top_n=[]
    for k in range(len(TopN)):
        n = TopN[k]
        print('N = ', n)
        items_index = []
        group_preference = []

        # for each candidate of item
        for i in range(len(Candidates)):

            # make permutation of user based on the intuitionistic order
            number_user = len(Users_index)
            user_permutation = Users_index
            print("original order ", user_permutation)
            for j in range(number_user):
                min = j
                for l in range(j, number_user):
                    order = Users_rating[min].getElement(Candidates[i]).order(Users_rating[l].getElement(Candidates[i]))
                    if order== 1:
                        min = l
                # swap
                temp = user_permutation[min]
                user_permutation[min] = user_permutation[j]
                user_permutation[j] = temp
            print('Current order/permutation', user_permutation)

            # calculate choquet integral-based aggregation
            weight = [None] * number_user
            for j in range(number_user):
                user_position = user_permutation[j]
                # print('user position ',user_position)
                temp_user_rating = OrgUserRating.iloc[user_position, 1:].dropna()
                # print('temp user rating', temp_user_rating)
                count = 0
                for l in range(temp_user_rating.shape[0]):
                    if (temp_user_rating.iloc[l] > 0):
                        count = count + 1
                weight[j] = count
                # weight[j]=Matrix[user_position,1:].select_dtypes(np.number).gt(0).sum(axis=1) #remove column of usesr id
            weight_prime = [None] * number_user
            for j in range(number_user):
                user_weight_prime = 0
                aveRate = OrgUserRating.iloc[user_permutation[j], 1:].median()
                temp_user_rating = OrgUserRating.iloc[user_permutation[j], 1:].dropna()
                for l in range(temp_user_rating.shape[0]):
                    if (temp_user_rating[l] >= aveRate):
                        user_weight_prime = user_weight_prime + 1
                    else:
                        user_weight_prime = user_weight_prime - 1
                weight_prime[j] = user_weight_prime
            # adjusted weight
            number_items = OrgUserRating.shape[1] - 1
            weight = np.true_divide(weight, number_items)
            weight_prime = np.true_divide(weight_prime, number_items)
            sumweight = np.sum(weight)
            weight = np.true_divide(weight, sumweight)

            def Capacity(Permuation, Index):
                '''return capacity of a subset form Index in a permuation'''
                capacity = 0
                #
                for m in range(Index, len(Permuation)):
                    capacity = capacity + weight[m]
                # if (Permuation.shape[0] - Index > 0):
                #     for m in range(Index, Permuation.shape[0]):
                #         capacity = capacity + weight_prime[m]
                if (len(Permuation) - Index > 0):
                    for m in range(Index, len(Permuation)):
                        capacity = capacity + weight_prime[m]
                if (capacity > 1):
                    return 1
                else:
                    return capacity
            # print( "capacity", Capacity(user_permutation, 3))

            # Fuzzy Choquet IF aggregation for group preference on item i
            #first element
            Capacity_gap = Capacity(user_permutation, 0) - Capacity(user_permutation, 1)
            group_item_pre = Users_rating[0].getElement(Candidates[i]).multiple(Capacity_gap)
            # group_item_pre is added by the next users' preference using sum function
            for j in range(1, len(Users_rating)):
                Capacity_gap = Capacity(user_permutation, j)-Capacity(user_permutation, j+1)
                group_item_pre = group_item_pre.sum(Users_rating[j].getElement(Candidates[i]).multiple(Capacity_gap))
            group_preference.append(group_item_pre)

        for i in range(len(group_preference)):
            if i < n:
                # items_index.append(i+1) # vi tri phan tu i trong UserRating la i+1
                items_index.append(i)
            else:
                min = 0
                for j in range(n):
                    if group_preference[items_index[min]].order(group_preference[items_index[j]]) == 1:
                        min = j
                if group_preference[items_index[min]].order(group_preference[i]) == -1:
                    items_index[min] = i
        print(' khuyen nghi ')
        print(items_index)
        top_n_items = []
        for i in range(len(items_index)):
            top_n_items.append(Candidates[items_index[i]])
        print(' top n ', top_n_items)

        metrics = get_metrics(Users_rating, temp_user_ma, top_n_items, n)
        print('performance ', metrics)
        metrics_top_n.append(metrics)

    # metrics for all elements in vector top_n
    metrics_top_n = pd.DataFrame(data=metrics_top_n)

    filename = RootFile + '_IF_Choquet_cap1_metrics.csv'
    metrics_top_n.to_csv(filename)
    return metrics_top_n

# least misery strategy
def IFS_least_misery_consensus (RootFile,Users_index, Users_rating, Candidates, TopN, OrgUserRating, temp_user_ma):
    # utily "Least misery consensus" on the IFS data set
    metrics_top_n = []
    for k in range(len(TopN)):
        n = TopN[k]
        print('N = ', n)
        items_index = []
        group_preference = []
        # for i in range(1,UserRating.shape[1]): # bo cot dau tien la chi so vi tri cua users
        for i in range(len(Candidates)):
            # chon dai dien là phan tu có order thấp nhất
            group_item_pre = Users_rating[0].getElement(Candidates[i])
            for j in range(1, len(Users_rating)):
                if group_item_pre.order(Users_rating[j].getElement(Candidates[i])) == 1: # current group_item_pre > Users_rating[j].getElement(Candidates[i])
                    group_item_pre = Users_rating[j].getElement(Candidates[i])
            group_preference.append(group_item_pre)
        # print(group_preference)
        # print(" user preference ",group_preference[0][0])
        for i in range(len(group_preference)):
            if i < n:
                # items_index.append(i+1) # vi tri phan tu i trong UserRating la i+1
                items_index.append(i)
            else:
                min = 0
                for j in range(n):
                    if group_preference[items_index[min]].order( group_preference[items_index[j]]) == -1:
                        min = j
                if group_preference[items_index[min]].order( group_preference[i]) == 1:
                    items_index[min] = i
        print(' khuyen nghi from Least misery strategy ')
        print(items_index)
        top_n_items = []
        for i in range(len(items_index)):
            top_n_items.append(Candidates[items_index[i]])
        print(' top n ', top_n_items)

        ''' get all metrics '''
        metrics = get_metrics(Users_rating, temp_user_ma, top_n_items, n)
        print('performance ', metrics)
        metrics_top_n.append(metrics)

        # metrics for all elements in vector top_n
    metrics_top_n = pd.DataFrame(data=metrics_top_n)

    filename = RootFile + '_IF_Least_Misery_metrics.csv'
    metrics_top_n.to_csv(filename)
    return metrics_top_n

# Most pleasure misery strategy
def IFS_most_pleasure_consensus (RootFile,Users_index, Users_rating, Candidates, TopN, OrgUserRating, temp_user_ma):

    metrics_top_n = []
    for k in range(len(TopN)):
        n = TopN[k]
        print('N = ', n)
        items_index = []
        group_preference = []
        # for i in range(1,UserRating.shape[1]): # bo cot dau tien la chi so vi tri cua users
        for i in range(len(Candidates)):
            # chon dai dien là phan tu có order cao nhất
            group_item_pre = Users_rating[0].getElement(Candidates[i])
            for j in range(1, len(Users_rating)):
                if group_item_pre.order(Users_rating[j].getElement(Candidates[i])) == -1:  # current group_item_pre > Users_rating[j].getElement(Candidates[i])
                    group_item_pre = Users_rating[j].getElement(Candidates[i])
            group_preference.append(group_item_pre)
        # print(group_preference)
        # print(" user preference ",group_preference[0][0])
        for i in range(len(group_preference)):
            if i < n:
                # items_index.append(i+1) # vi tri phan tu i trong UserRating la i+1
                items_index.append(i)
            else:
                min = 0
                for j in range(n):
                    if group_preference[items_index[min]].order(group_preference[items_index[j]]) == -1:
                        min = j
                if group_preference[items_index[min]].order(group_preference[i]) == 1:
                    items_index[min] = i
        print(' khuyen nghi from Most Pleasure strategy')
        print(items_index)
        top_n_items = []
        for i in range(len(items_index)):
            top_n_items.append(Candidates[items_index[i]])
        print(' top n ', top_n_items)
        ''' get all metrics '''
        ''' get all metrics '''
        metrics = get_metrics(Users_rating, temp_user_ma, top_n_items, n)
        print('performance ', metrics)
        metrics_top_n.append(metrics)

        # metrics for all elements in vector top_n
    metrics_top_n = pd.DataFrame(data=metrics_top_n)

    filename = RootFile + '_IF_most_pleasure_metrics.csv'
    metrics_top_n.to_csv(filename)
    return metrics_top_n

# Approval voting strategy
def IFS_approval_voting_consensus (RootFile,Users_index, Users_rating, Candidates, TopN, OrgUserRating, temp_user_ma):

    metrics_top_n = []
    #define voted threshold
    voted_threshold = IFE.IFE()
    voted_threshold.setMembership(0.65)
    voted_threshold.setNonmembership(0.1)
    voted_threshold.setHesitance(0.25)
    ################################################
    for k in range(len(TopN)):
        n = TopN[k]
        print('N = ', n)
        items_index = []
        group_preference = []
        # for i in range(1,UserRating.shape[1]): # bo cot dau tien la chi so vi tri cua users
        for i in range(len(Candidates)):
            group_item_pre=0
            for j in range(len(Users_rating)):
                if Users_rating[j].getElement(Candidates[i]).order(voted_threshold)>=0:
                    group_item_pre = group_item_pre + 1
            group_preference.append(group_item_pre)
        # print(group_preference)
        # print(" user preference ",group_preference[0][0])
        for i in range(len(group_preference)):
            if i < n:
                # items_index.append(i+1) # vi tri phan tu i trong UserRating la i+1
                items_index.append(i)
            else:
                min = 0
                for j in range(n):
                    if group_preference[items_index[min]] > group_preference[items_index[j]]:
                        min = j
                if group_preference[items_index[min]] < group_preference[i]:
                    items_index[min] = i
        print(' khuyen nghi from Approval voting strategy')
        print(items_index)
        top_n_items = []
        for i in range(len(items_index)):
            top_n_items.append(Candidates[items_index[i]])
        print(' top n ', top_n_items)
        ''' get all metrics '''
        metrics = get_metrics(Users_rating, temp_user_ma, top_n_items, n)
        print('performance ', metrics)
        metrics_top_n.append(metrics)

        # metrics for all elements in vector top_n
    metrics_top_n = pd.DataFrame(data=metrics_top_n)

    filename = RootFile + '_IF_approval_voting_metrics.csv'
    metrics_top_n.to_csv(filename)
    return metrics_top_n

# Copeland rule strategy
def IFS_Copeland_rule_consensus (RootFile,Users_index, Users_rating, Candidates, TopN, OrgUserRating, temp_user_ma):

    metrics_top_n=[]

    ################################################
    group_preference = []
    # for i in range(1,UserRating.shape[1]): # bo cot dau tien la chi so vi tri cua users
    for i in range(len(Candidates)):
        group_item_pre = 0
        for j in range(len(Candidates)):
            if i!=j:
                user_refer_a = 0
                user_refer_b = 0
                for k in range(len(Users_rating)):
                    if Users_rating[k].getElement(Candidates[i]).order(Users_rating[k].getElement(Candidates[j]))==1:
                        user_refer_a = user_refer_a +1
                    else:
                        if Users_rating[k].getElement(Candidates[i]).order(Users_rating[k].getElement(Candidates[j]))==0:
                            user_refer_a = user_refer_a + 0.5
                            user_refer_b=user_refer_b+0.5
                        else:
                            user_refer_b = user_refer_b + 1
                if user_refer_a > user_refer_b:
                    group_item_pre = group_item_pre+1
                else:
                    if user_refer_a == user_refer_b:
                        group_item_pre = group_item_pre + 0.5
        group_preference.append(group_item_pre)
    # print(group_preference)
    for k in range(len(TopN)):
        n = TopN[k]
        print('N = ',n)
        items_index=[]
        for i in range(len(group_preference)):
            if i < n:
                # items_index.append(i+1) # vi tri phan tu i trong UserRating la i+1
                items_index.append(i)
            else:
                min = 0
                for j in range(n):
                    if group_preference[items_index[min]] > group_preference[items_index[j]]:
                        min = j
                if group_preference[items_index[min]] < group_preference[i]:
                    items_index[min] = i
        print(' khuyen nghi from Copeland rule strategy: ')
        print(items_index)
        top_n_items = []
        for i in range(len(items_index)):
            top_n_items.append(Candidates[items_index[i]])
        print(' top n ', top_n_items)
        ''' get all metrics '''
        metrics = get_metrics(Users_rating, temp_user_ma, top_n_items, n)
        print('performance ', metrics)
        metrics_top_n.append(metrics)

        # metrics for all elements in vector top_n
    metrics_top_n = pd.DataFrame(data=metrics_top_n)
    filename = RootFile + '_IF_Copeland_rule_metrics.csv'
    metrics_top_n.to_csv(filename)
    return metrics_top_n


def DIFGRS_test():
    size = [3, 5, 10, 15, 20]
    for i in range(len(size)):
        group_size = size[i]
        group_number = 30
        statisticfile = 'DIFGRS_results\\statistic\\' + str(group_size) + '_group_rating_'
        df_AS = pd.DataFrame(0, index=np.arange(10), columns=np.arange(4))
        df_LM = pd.DataFrame(0, index=np.arange(10), columns=np.arange(4))
        df_MP = pd.DataFrame(0, index=np.arange(10), columns=np.arange(4))
        df_CR = pd.DataFrame(0, index=np.arange(10), columns=np.arange(4))
        df_AV = pd.DataFrame(0, index=np.arange(10), columns=np.arange(4))
        df_CS = pd.DataFrame(0, index=np.arange(10), columns=np.arange(4))
        for j in range(group_number):
            filename = 'DIFGRS_results\\group_rating_3-5-23\\' + str(group_size) + '_group_rating_' + str(j) + '.csv'
            # filename = '3_group_rating_0.csv'
            print(" Filename: ", filename)
            rootfile='DIFGRS_results\\statistic_3-5-23_v2\\' + str(group_size) + '_group_rating_' + str(j)
            ''' test for each rating file'''
            # get user predicted rates
            users_rate = []
            users_index = []
            df = pd.read_csv(filename)
            numberofmembers = int(df.shape[0] / 4) #thông tin về một user được lưu trên 4 dòng (timed-IF)
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

            # get original information to estimate measurement metrics
            defuzzi_matrix = []
            for i in range(0, len(users_rate)):
                user_index = users_index[i]  # keep the index of user in original rating matrix
                user_rate_defuzzification = []
                user_rate_defuzzification.append(user_index)
                items_number = users_rate[i].getNumber_elements()
                print("number ", items_number)
                for j in range(items_number):
                    obj = users_rate[i].getElement(j)
                    defuzzilier = IFE_fuzzilizer.IFE_defuzzilier()
                    # obj.show()
                    # user_item_rate_deffuzzification = defuzzilier.getScriptation(obj)
                    # user_item_rate_deffuzzification = defuzzilier.getStandardScriptation(obj)
                    user_item_rate_deffuzzification = defuzzilier.getAdjustStandardScriptation(obj)
                    # print("scrip value ", user_item_rate_deffuzzification)
                    # add to list of userate
                    user_rate_defuzzification.append(user_item_rate_deffuzzification)
                defuzzi_matrix.append(user_rate_defuzzification)
            # get all metrics
            temp_user_ma = pd.DataFrame(defuzzi_matrix)
            temp_user_ma.to_csv("temp_defuzzy.csv")
            # original rates
            OrgUserRating = pd.read_csv("movielens.csv")

            ''' run needed algorithms  '''
            print(" FUZZY RECOMMENDATION")
            print(" cac users duoc du bao")
            print(users_rate)
            print(" users' index ", users_index)

            individual_can = 150
            candidates = get_individual_recommendation(users_rate, individual_can)
            print(" list of candidates")
            print(candidates)
            print(" total number ", len(candidates))

            # top_n = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
            top_n = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 20, 25, 30, 35, 40]
            #additive consensus
            df_AS_i= IFS_additive_consensus(rootfile, users_index, users_rate, candidates, top_n, OrgUserRating, temp_user_ma)
            # print("diaf",df_AS_i)
            df_AS= df_AS.add(df_AS_i, fill_value=0)
            # multiple consensus
            # IFS_multiple_consensus(rootfile, users_index, users_rate, candidates, top_n, OrgUserRating, temp_user_ma)
            # other fuzzy Consensus
            df_LM_i = IFS_least_misery_consensus(rootfile, users_index, users_rate, candidates, top_n, OrgUserRating, temp_user_ma)
            df_LM= df_LM.add(df_LM_i, fill_value=0)

            df_MP_i= IFS_most_pleasure_consensus(rootfile, users_index, users_rate, candidates, top_n, OrgUserRating, temp_user_ma)
            df_MP=df_MP.add(df_MP_i, fill_value=0)

            df_AV_i= IFS_approval_voting_consensus(rootfile, users_index, users_rate, candidates, top_n, OrgUserRating, temp_user_ma)
            df_AV=df_AV.add(df_AV_i, fill_value=0)

            df_CR_i= IFS_Copeland_rule_consensus(rootfile, users_index, users_rate, candidates, top_n, OrgUserRating, temp_user_ma)
            df_CR= df_CR.add(df_CR_i, fill_value=0)
            # Fuzzy Choquet integral based aggregation Consensus
            df_CS_i= IFS_Choquet_aggregation_consensus (rootfile,users_index, users_rate, candidates, top_n, OrgUserRating, temp_user_ma)
            df_CS=df_CS.add(df_CS_i,fill_value=0)

        df_AS=df_AS.div(group_number)
        adfile=statisticfile+"aditive.csv"
        df_AS.to_csv(adfile)
        df_LM=df_LM.div(group_number)
        lmFIle = statisticfile+"leastmeasery.csv"
        df_LM.to_csv(lmFIle)
        df_MP=df_MP.div(group_number)
        mpFile =statisticfile +"mostpleasure.csv"
        df_MP.to_csv(mpFile)
        df_AV=df_AV.div(group_number)
        avFIle = statisticfile+"aproval_voting.csv"
        df_AV.to_csv(avFIle)
        df_CR=df_CR.div(group_number)
        crFile =statisticfile+"coplandRule.csv"
        df_CR.to_csv(crFile)
        df_CS=df_CS.div(group_number)
        csFile=statisticfile+"ChoquetAgg.csv"
        df_CS.to_csv(csFile)


if __name__ == '__main__':
    DIFGRS_test()
