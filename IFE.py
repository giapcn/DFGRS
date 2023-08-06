''' create by Giap '''

import numpy as np
import math

#intuitionistic fuzzy element
class IFE:
    def __init__(self,Membership=0,Nonmembership=1,Hesitance=0):
        self.membership = Membership
        self.nonmembership = Nonmembership
        self.hesitance = Hesitance

    def getMembership(self):
        return self.membership
    def getNonmembership(self):
        return self.nonmembership
    def getHesitance(self):
        return self.hesitance

    def setMembership(self, Mebership):
        self.membership = Mebership

    def setNonmembership(self, Nonmembership):
        self.nonmembership = Nonmembership

    def setHesitance(self, Hesitance):
        self.hesitance = Hesitance


    #scores and order
    def score(self):
        score= self.membership-self.nonmembership
        return score
    def accuracy(self):
        accuracy= self.membership+ self.nonmembership
        return accuracy

    #an order between two IF numbers
    def order(self, other):
        assert isinstance(other, IFE)
        order = 0 #equal
        if self.score()<other.score():
            order = -1 # smaller
        else:
            if self.score()==other.score():
                if self.accuracy() <other.accuracy():
                    order = -1
                if self.accuracy() ==other.accuracy():
                    order=0
                if self.accuracy() > other.accuracy():
                    order=1
            else:
                order = 1 #greater
        return order

    #sum and product operations
    def sum(self, other):
        assert isinstance(other, IFE)
        Membership = self.membership + other.membership - np.dot(self.membership, other.membership)
        Nonmembership = np.dot(self.nonmembership, other.nonmembership)
        result = IFE(Membership, Nonmembership, 1-Membership-Nonmembership)
        return result

    def product(self, other):
        assert isinstance(other, IFE)
        Membership = np.dot(self.membership,other.membership)
        Nonmembership = self.nonmembership+other.nonmembership - np.dot(self.nonmembership, other.nonmembership)
        result = IFE(Membership,Nonmembership,1-Membership-Nonmembership)
        return result

    # subtract operation
    def subtract(self, other):
        assert isinstance(other, IFE)
        Membership = 0;
        if other.membership<1:
            Mem_buff = (self.membership-other.membership)/(1 - other.membership)
            if Mem_buff > 0:
                Membership = Mem_buff
        Nonmembership=1
        if other.nonmembership ==0:
            Nonmembership=1
        else:
            Non_buff = self.nonmembership/ other.nonmembership
            if Non_buff > (1-self.membership)/(1-other.membership):
                Non_buff = (1 - self.membership) / (1 - other.membership)
            if Nonmembership > Non_buff:
                Nonmembership= Non_buff
        # return result
        Membership = self.membership + other.membership - np.dot(self.membership, other.membership)
        Nonmembership = np.dot(self.nonmembership, other.nonmembership)
        result = IFE(Membership, Nonmembership, 1 - Membership - Nonmembership)
        return result

    # multiple to constance
    def multiple(self, lamda):
        Membership = 1 - np.power(1-self.membership, lamda)
        Nonmembership = np.power(self.nonmembership, lamda)
        result = IFE(Membership, Nonmembership, 1 - Membership - Nonmembership)
        return result

    # power operation
    def power(self, lamda):
        Membership = np.power(self.membership, lamda)
        Nonmembership = 1-np.power(1 - self.nonmembership, lamda)
        result = IFE(Membership, Nonmembership, 1 - Membership - Nonmembership)
        return result

    #other operations
    def isvalidate(self):
        if self.membership+ self.nonmembership+ self.hesitance ==1:
            if(self.membership>=0 & self.nonmembership>=0 & self.hesitance):
                return True
            else:
                return False
        else:
            return False

    def show(self):
        print("Membership= ",self.membership)
        print("Nonmembership= ", self.nonmembership)
        print("Hesistance= ", self.hesitance)

    #distance between two elements
    def disHamming(self,other):
        assert isinstance(other, IFE)
        difs = abs(self.membership-other.membership)+ abs(self.nonmembership-other.nonmembership)+ abs(self.hesitance-other.hesitance)
        return difs/2
    def disEuclidean(self,other):
        assert isinstance(other, IFE)
        difs = pow(self.membership-other.membership,2)+ pow(self.nonmembership-other.nonmembership,2)+ pow(self.hesitance-other.hesitance,2)
        return (math.sqrt(difs))/2

    def disHammingModify(self,other):
        assert isinstance(other, IFE)
        difs = max(abs(self.membership-other.membership), abs(self.nonmembership-other.nonmembership))
        return difs

    def disEuclideanModify(self,other):
        assert isinstance(other, IFE)
        difs = max(pow(self.membership-other.membership,2), pow(self.nonmembership-other.nonmembership,2))
        return math.sqrt(difs)

    # define a distance used to calculate the accurarcy of a prediction algorithm
    def disAccScore(self,other):
        assert isinstance(other, IFE)
        difs = pow(self.membership-other.membership,2)+ pow(self.nonmembership-other.nonmembership,2)+ pow(self.hesitance-other.hesitance,2)
        return math.sqrt(difs/3)