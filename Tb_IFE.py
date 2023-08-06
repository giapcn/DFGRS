''' create by Giap '''

import numpy as np
import pandas as pd
import math
import IFE
import time_convertion

#time based intuitionistic fuzzy element/vector
class Tb_IFvector:
    def __init__(self, initialEles=0):
        self.eles_num = initialEles
        self.membership = np.array((1,self.eles_num),dtype=float)
        self.nonmembership = np.array((1,self.eles_num),dtype=float)
        self.hesitance = np.array((1,self.eles_num),dtype=float)
        self.timepoint = np.zeros((1,self.eles_num),dtype=float) #in decay of time
        #define the max distance value
        self.UnknowDis =1

    def show(self):
        print(self.membership)
        print(self.nonmembership)
        print(self.hesitance)
        print(self.timepoint)

    def save(self, Filename):

        dataset = pd.DataFrame({'membership': self.membership, 'nonmembership': self.nonmembership,\
                                'hesitance':self.hesitance, 'timestamp' : self.timepoint}, \
                               columns=['membership', 'nonmembership','hesitance','timestamp'])
        dataset.to_csv(Filename)
    #get functions
    def getMembership(self):
        return self.membership
    def getNonMembership(self):
        return self.nonmembership
    def getHesitance(self):
        return self.hesitance
    def getTimepoint(self):
        return self.timepoint

    #get element by index
    def getElement(self, index):
        if np.isnan(self.membership[index]):
            return None
        else:
            result = IFE.IFE()
            result.membership= self.membership[index]
            result.nonmembership= self.nonmembership[index]
            result.hesitance= self.hesitance[index]
            return result

    # get timepoint by index
    def getTimepoint_by_index(self, index):
        if np.isnan(self.membership[index]):
            return None
        else:
            result = self.timepoint[index]
            return result

    # set specific element
    def setElement(self, index, Membership, NonMembership, Hesitance, Timepoint):
        # if np.isnan(self.membership[index]):
        #     self.membership[index]= Membership
        #     self.nonmembership[index] = NonMembership
        #     self.hesitance[index] = Hesitance
        #     self.timepoint[index] = Timepoint

        self.membership[index]= Membership
        self.nonmembership[index] = NonMembership
        self.hesitance[index] = Hesitance
        self.timepoint[index] = Timepoint

    # set the list of elements
    def setEles(self, Membership, NonMembership,Hesitance,TimePoint):
        #Membership, NonMembership,Hesitance,TimePoint are numpy arrays
        self.membership = np.copy(Membership)
        self.nonmembership = np.copy(NonMembership)
        self.hesitance = np.copy(Hesitance)
        self.timepoint = np.copy(TimePoint)
        self.eles_num = len(self.membership)

    def getNumber_elements(self):
        return self.eles_num

    def getNumber_NotNA_elements(self):
        na_mems = np.isnan(self.membership)
        not_na_mems = ~na_mems
        valid_membership = self.membership[not_na_mems]
        valid_nonmembership = self.nonmembership[not_na_mems]
        n = len(valid_membership)
        return n

    # check the null value
    def isNull(self, index):
        if np.isnan(self.membership[index]):
            return True
        else:
            return False

    #similarity of two TB_IFvectors
    def similarity(self, other, sim_type="cosine"):
        assert isinstance(other, Tb_IFvector)
        #identify the valid vector
        na_mems = np.isnan(self.membership)
        na_other_mems = np.isnan(other.membership)
        valid = (~na_mems)&(~na_other_mems)
        #print(valid)
        if sim_type == "cosine":
            cos_sim = 0
            valid_number = 0
            n=len(self.membership)
            for i in range(n):
                #print("i =", i)
                if valid[i]:
                    valid_number += 1
                    numerator = np.dot(self.membership[i], other.membership[i])
                    numerator += np.dot(self.nonmembership[i], other.nonmembership[i])
                    denominator = math.sqrt(pow(self.membership[i],2)+ pow( self.nonmembership[i],2))
                    denominator *= math.sqrt(pow(other.membership[i],2)+ pow( other.nonmembership[i],2))
                    #print(" mau ",denominator)
                    if denominator>0:
                        cos_sim += numerator/denominator
            if valid_number == 0:
                return self.UnknowDis
            else:
                return cos_sim/valid_number
        else:
            if sim_type == "pearson":
                correlation = 0
                n = len(self.membership)
                selfmem_mean = self.membership.mean()
                selfnon_mean = self.nonmembership.mean()
                selfhes_mean = self.hesitance.mean()
                othermem_mean = other.membership.mean()
                othernon_mean = other.nonmembership.mean()
                otherhes_mean = other.hesitance.mean()
                r1_numerator=0
                r1_denominator_p1=0
                r1_denominator_p2=0
                r2_numerator = 0
                r2_denominator_p1 = 0
                r2_denominator_p2 = 0
                r3_numerator = 0
                r3_denominator_p1 = 0
                r3_denominator_p2 = 0
                valid_number=0
                for i in range(n):
                    if valid[i]:
                        valid_number+=1
                        #r1
                        r1_numerator += np.dot(self.membership[i]-selfmem_mean, other.membership[i]-othermem_mean)
                        r1_denominator_p1 += pow(self.membership[i]-selfmem_mean, 2)
                        r1_denominator_p1 += pow(other.membership[i]-othermem_mean, 2)
                        #r2
                        r2_numerator += np.dot(self.nonmembership[i] - selfnon_mean, other.nonmembership[i] - othernon_mean)
                        r2_denominator_p1 += pow(self.nonmembership[i] - selfnon_mean, 2)
                        r2_denominator_p1 += pow(other.nonmembership[i] - othernon_mean, 2)
                        # r3
                        r3_numerator += np.dot(self.hesitance[i] - selfhes_mean, other.hesitance[i] - otherhes_mean)
                        r3_denominator_p1 += pow(self.hesitance[i] - selfhes_mean, 2)
                        r3_denominator_p1 += pow(other.hesitance[i] - otherhes_mean, 2)
                r1 = r1_numerator/(math.sqrt(r1_denominator_p1)*math.sqrt(r1_denominator_p2))
                r2 = r2_numerator / (math.sqrt(r2_denominator_p1) * math.sqrt(r2_denominator_p2))
                r3 = r3_numerator / (math.sqrt(r3_denominator_p1) * math.sqrt(r3_denominator_p2))
                return ((r1+r2+r3)/3)/valid_number
            else:
                return None
    # C_similarity
    def X_similarity(self, other,sim_type="C",power=2):
        assert isinstance(other, Tb_IFvector)
        # identify the valid vector
        na_mems = np.isnan(self.membership)
        na_other_mems = np.isnan(other.membership)
        valid = (~na_mems) & (~na_other_mems)
        print(valid)
        # calculate similarity
        if sim_type == "C":
            C_sim=0
            n = len(self.membership)
            for i in range(n):
                if valid[i]:
                    C_sim += abs(abs(self.membership[i]-self.nonmembership[i])-abs(other.membership[i]-other.nonmembership[i]))
            return 1-C_sim/2*n
        if sim_type == "DC":
            DC_sim=0
            n = len(self.membership)
            for i in range(n):
                if valid[i]:
                    self_psi = (self.membership[i]+ 1-self.nonmembership[i])/2
                    other_psi = (other.membership[i]+ 1-other.nonmembership[i])/2
                    DC_sim += pow(abs(self_psi- other_psi),power)
            return 1- (DC_sim/n)**(1/float(power))
        if sim_type == "H":
            H_sim=0
            n = len(self.membership)
            for i in range(n):
                if valid[i]:
                    H_sim += abs(self.membership[i]-other.membership[i])+ abs(other.nonmembership[i]-other.nonmembership[i])
            return 1-H_sim/2*n
        if sim_type == "L":
            C_sim=0
            H_sim=0
            n = len(self.membership)
            for i in range(n):
                if valid[i]:
                    C_sim += abs(abs(self.membership[i]-self.nonmembership[i])-abs(other.membership[i]-other.nonmembership[i]))
                    H_sim += abs(self.membership[i]-other.membership[i])+ abs(other.nonmembership[i]-other.nonmembership[i])
            L_sim= 1-C_sim/4*n -H_sim/4*n
            return L_sim
        else:
            return None
    # similarity of two TB_IFvectors using distance based on Hausdorff metric
    def H_similarity(self, other,sim_type="HY1",power=2):
        assert isinstance(other, Tb_IFvector)
        # identify the valid vector
        na_mems = np.isnan(self.membership)
        na_other_mems = np.isnan(other.membership)
        valid = (~na_mems) & (~na_other_mems)
        print(valid)
        # calculate similarity
        if sim_type == "HY1": #Hausdorff - hamming distance
            HY_sim = 0
            n = len(self.membership)
            for i in range(n):
                if valid[i]:
                    HY_sim += max(abs(self.membership[i]-other.membership[i]), abs(self.nonmembership[i]-other.nonmembership[i]))
            return 1-HY_sim/n
        if sim_type == "HY2": #Hausdorff - euclidean distance
            HY_sim = 0
            n = len(self.membership)
            for i in range(n):
                if valid[i]:
                    HY_sim += max(abs(self.membership[i] - other.membership[i]),
                                  abs(self.nonmembership[i] - other.nonmembership[i]))
            HY_sim= HY_sim/n
            return (math.exp(-HY_sim)-math.exp(-1))/(1- math.exp(-1))
        if sim_type == "HY3":  # Hausdorff - euclidean distance: a modification
            HY_sim = 0
            n = len(self.membership)
            for i in range(n):
                if valid[i]:
                    HY_sim += max(abs(self.membership[i] - other.membership[i]),
                                  abs(self.nonmembership[i] - other.nonmembership[i]))
            HY_sim = HY_sim/n
            return (1 - HY_sim)/(1 + HY_sim)
        else:
            return None

    # similarity of two TB_IFvectors with out time effect
    def similarity_with_TE(self, other, time_Frequency=86400, time_Lamda=0.001):
        assert isinstance(other, Tb_IFvector)
        # identify the valid vector
        na_mems = np.isnan(self.membership)
        na_other_mems = np.isnan(other.membership)
        valid = (~na_mems) & (~na_other_mems)
        print(valid)
        # calculate similarity
        tb_cos_sim = 0
        valid_num=0
        tc = time_convertion.time_convertion(Root= 0, Frequency=time_Frequency, Lamda= time_Lamda) #using specific time effect (Frequency and parameter Lamda)
        for i in range(len(self.membership)):
            if valid[i]:
                valid_num += 1
                numerator = np.dot(self.membership[i], other.membership[i])
                numerator += np.dot(self.nonmembership[i], other.nonmembership[i])

                denominator = max(np.dot(self.membership[i], self.membership[i]),
                                   np.dot(other.membership[i], other.membership[i]))
                denominator += max(np.dot(self.nonmembership[i], self.nonmembership[i]),
                                   np.dot(other.nonmembership[i], other.nonmembership[i]))
                time_eff = tc.time_effect(self.timepoint[i], other.timepoint[i])
                if denominator>0:
                    tb_cos_sim+= (numerator / denominator)*time_eff
        if valid_num == 0:
            return self.UnknowDis
        else:
            return tb_cos_sim/valid_num

    # # similarity of two TB_IFvectors with weighted vector
    # def weighted_similarity(self, other, weights):
    #     assert isinstance(other, Tb_IFvector)
    #     numerator = 0
    #     denominator = 0
    #     for i in range(len(self.membership)):
    #         numerator += weights[i] * np.dot(self.membership[i], other.membership[i])
    #         numerator += weights[i] * np.dot(self.non_membership[i], other.non_membership[i])
    #
    #         denominator += max(np.dot(self.membership[i], self.membership[i]), np.dot(other.membership[i], other.membership[i]))
    #         denominator += max(np.dot(self.non_membership[i], self.non_membership[i]), np.dot(other.non_membership[i], other.non_membership[i]))
    #     return numerator/denominator/np.sum(weights)
    #
    # def distance(self, other, dist_type="euclidean"):
    #     def distance(disease, patient, dist_type):
    #         result = 0
    #         if dist_type == "euclidean":
    #             for i in range(len(patient)):
    #                 result += (disease[i][0] - patient[i][0])**2+(disease[i][1] - patient[i][1])**2 + ((1 - (patient[i][0] + patient[i][1])) - (1 - (disease[i][0] + disease[i][1])))**2
    #             return np.sqrt(result * 1/(len(patient)*2))
    #         if dist_type == "absolute":
    #             for i in range(len(patient)):
    #                 result += np.abs(disease[i][0] - patient[i][0]) + np.abs(disease[i][1] - patient[i][1]) + np.abs((1 - (patient[i][0] + patient[i][1])) - (1 - (disease[i][0] + disease[i][1])))
    #             return result / (len(patient)*2)
    #
    #     if dist_type not in ["euclidean", "absolute"]:
    #         print("Wrong distance type, using euclidean instead.")
    #         dist_type = "euclidean"
    #     assert isinstance(other, Tb_IFvector)
    #     numerator = 0
    #     denominator = 0
    #     for i in range(len(self.membership)):
    #         numerator += np.dot(self.membership[i], other.membership[i])
    #         numerator += np.dot(self.non_membership[i], other.non_membership[i])
    #
    #         denominator += max(np.dot(self.membership[i], self.membership[i]),
    #                            np.dot(other.membership[i], other.membership[i]))
    #         denominator += max(np.dot(self.non_membership[i], self.non_membership[i]),
    #                            np.dot(other.non_membership[i], other.non_membership[i]))
    #     return numerator / denominator

    #define IAFW operations
    def IAFW(self,Weights):
        #notice: Weights is a numpy array
        # Membership = 1 - np.product(np.dot(1 - self.membership, Weights))
        # Nonmembership = np.product(np.dot(self.nonmembership, Weights))
        # identify the valid vector (remove NaN values
        na_mems = np.isnan(self.membership)
        not_na_mems = ~na_mems
        valid_membership = self.membership[not_na_mems]
        valid_nonmembership = self.nonmembership[not_na_mems]
        valid_weights = Weights[not_na_mems]
        Membership=1- np.product(np.power(1-valid_membership,valid_weights))
        Nonmembership = np.product(np.power(valid_nonmembership,valid_weights))
        #create a intuitionistic fuzzy number
        IAFW_ins = IFE.IFE(Membership,Nonmembership,1-Membership-Nonmembership)
        return IAFW_ins

    # define IAFW_mean operations
    def IAFW_mean(self):
        # notice: this mean using equal weights= 1/n (n= valid element)
        # Membership = 1 - np.product(np.dot(1 - self.membership, Weights))
        # Nonmembership = np.product(np.dot(self.nonmembership, Weights))
        # identify the valid vector (remove NaN values
        na_mems = np.isnan(self.membership)
        not_na_mems = ~na_mems
        valid_membership = self.membership[not_na_mems]
        valid_nonmembership = self.nonmembership[not_na_mems]
        n = len(valid_membership)
        valid_weights = np.empty(n)
        valid_weights.fill(1/n)
        #test
        # print('N =',n)
        # print("membership")
        # print(valid_membership)
        # print("non-membership")
        # print(valid_nonmembership)
        # print(valid_weights)
        Membership = 1 - np.product(np.power(1 - valid_membership, valid_weights))
        Nonmembership = np.product(np.power(valid_nonmembership, valid_weights))
        # create a intuitionistic fuzzy number
        if Membership==1:
            print("hahaha")
        IAFW_mean = IFE.IFE(Membership, Nonmembership, 1 - Membership - Nonmembership)
        return IAFW_mean
    # define Bonfferroni_mean operations
    def Bonfferroni_mean(self, p=1, q=1):
        # notice: using in different way as mean using IAFW
        # identify the valid vector (remove NaN values
        na_mems = np.isnan(self.membership)
        not_na_mems = ~na_mems
        valid_membership = self.membership[not_na_mems]
        valid_nonmembership = self.nonmembership[not_na_mems]
        n= len(valid_membership)
        print('valid value rates: ',n)
        if n<=0:
            return None
        elif n ==1:
            Membership= valid_membership[0]
            Nonmembership= valid_nonmembership[0]
            BM_ins = IFE.IFE(Membership, Nonmembership, 1 - Membership - Nonmembership)
            return BM_ins
        else:
            power_thr_1 = 1/(n*(n-1))
            power_thr_2 = 1/(p + q)
            Membership=1
            Nonmembership=1
            for i in range(len(valid_membership)):
                for j in range(len(valid_membership)):
                    if i != j:
                        Membership *= np.power(1 - np.dot(np.power(valid_membership[i],p), np.power(valid_membership[j],q)),power_thr_1)
                        Nonmembership *= np.power(1- np.dot(np.power(1-valid_nonmembership[i],p), np.power(1-valid_nonmembership[j],q)),power_thr_1)
            Membership = np.power(1-Membership,power_thr_2)
            Nonmembership=1- np.power(1-Nonmembership, power_thr_2)
            # create a Bonfferroni mean of an a vector of IF elements(intuitionistic fuzzy number)
            BM_ins = IFE.IFE(Membership, Nonmembership, 1 - Membership - Nonmembership)
            return BM_ins

    def countAvailable(self):
        # notice: this function count number of available elements
        # identify the valid vector (remove NaN values
        na_mems = np.isnan(self.membership)
        not_na_mems = ~na_mems
        valid_membership = self.membership[not_na_mems]
        valid_nonmembership = self.nonmembership[not_na_mems]
        n = len(valid_membership)
        return n

    def countGreaterAverage(self):
        # notice: this function count number of elements that greater or equal to mean

        aveRate = self.IAFW_mean()
        # identify the valid vector (remove NaN values
        na_mems = np.isnan(self.membership)
        not_na_mems = ~na_mems
        valid_membership = self.membership[not_na_mems]
        valid_nonmembership = self.nonmembership[not_na_mems]
        n = len(valid_membership)
        count =0
        for i in range(len(valid_membership)):
            curObj = IFE.IFE(valid_membership[i],valid_nonmembership[i], 1-valid_membership[i]-valid_nonmembership[i])
            if(curObj.order(aveRate)!= -1):
                count=count+1
        return count

    def countWeight_prime(self):
        # notice: this function count number of elements that greater or equal to mean

        aveRate = self.IAFW_mean()
        # identify the valid vector (remove NaN values
        na_mems = np.isnan(self.membership)
        not_na_mems = ~na_mems
        valid_membership = self.membership[not_na_mems]
        valid_nonmembership = self.nonmembership[not_na_mems]
        n = len(valid_membership)
        count = 0
        for i in range(len(valid_membership)):
            curObj = IFE.IFE(valid_membership[i], valid_nonmembership[i],
                             1 - valid_membership[i] - valid_nonmembership[i])
            if curObj.order(aveRate) == -1:
                count = count + 1
            else:
                if curObj.order(aveRate) == 1:
                    count = count - 1

        return count



if __name__ =='__main__':
    #using for test
    print("testing IF means")
    membership = np.array([0.4, 0.5, 0.6])
    nonmembership = np.array([0.4, 0.1, 0.2])
    hesistant = 1 - membership - nonmembership
    timepoint =np.array([4, 15, 60])
    ins =Tb_IFvector()
    ins.setEles(membership,nonmembership,hesistant,timepoint)
    iafw_result = ins.IAFW_mean()
    print("IAFW mean", iafw_result.membership, iafw_result.nonmembership, iafw_result.hesitance)
    boffe_mean = ins.Bonfferroni_mean()
    print("Bofferroni mean", boffe_mean.membership, boffe_mean.nonmembership,boffe_mean.hesitance)

    boffe_mean = ins.Bonfferroni_mean(1, 2)
    print("Bofferroni mean", boffe_mean.membership, boffe_mean.nonmembership, boffe_mean.hesitance)
    boffe_mean = ins.Bonfferroni_mean(2, 3)
    print("Bofferroni mean", boffe_mean.membership, boffe_mean.nonmembership, boffe_mean.hesitance)
    boffe_mean = ins.Bonfferroni_mean(1, 10)
    print("Bofferroni mean", boffe_mean.membership, boffe_mean.nonmembership, boffe_mean.hesitance)
    boffe_mean = ins.Bonfferroni_mean(1, 20)
    print("Bofferroni mean", boffe_mean.membership, boffe_mean.nonmembership, boffe_mean.hesitance)