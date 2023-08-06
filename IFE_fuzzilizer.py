''' create by Giap '''

import math
import copy
import IFE

#intuitionistic fuzzilizer
class IFE_fuzzilier:
    def __init__(self,Uth=5.5,Lth=0,A1=1,B1=5.5,C1=0.5,A2=1,B2=0,C2=0.25):
        self.uth = Uth
        self.lth = Lth
        self.a1 = A1
        self.b1=B1
        self.c1=C1
        self.a2 = A2
        self.b2 = B2
        self.c2 = C2

    #mu
    def mu(self, X):
        # score = self.a1 * math.exp(-(math.pow(X - self.b1, 2) / 2 * self.c1))
        # return score
        if X>=self.uth:
            return 1
        else:
            if X<=self.lth:
                return 0
            else:
                score= self.a1*math.exp(-(math.pow(X-self.b1,2)/2*self.c1))
                return score

    # nu
    def nu(self, X):
        if X >= self.uth:
            return 0
        else:
            if X <= self.lth:
                return 1
            else:
                score = self.a2 * math.exp(-(math.pow(X - self.b2, 2) / 2 * self.c2))
                return score
    #pi + validation
    def pi(self, X):
        pi = 1- self.mu(X) - self.nu(X)
        if pi>1:
            print(" Error. Data validation.")
        else:
            if pi<0:
                print(" Error. Data validation.")
            else:
                return pi

#intuitionistic defuzzilizer
class IFE_defuzzilier:
    def __init__(self,Uth=5.5,Lth=0,A1=1,B1=5.5,C1=0.5):
        self.uth = Uth
        self.lth = Lth
        self.a1 = A1
        self.b1=B1
        self.c1=C1

    #get scriptation
    def getScriptation(self, Obj):
        Mu = Obj.getMembership()
        # print(" mu = ",Mu)
        # return defuzzification value bassed on membership value
        anpha = -(self.c1/2)
        beta = self.b1*self.c1
        lamda = - (math.pow(self.b1,2)* self.c1 )/2 + math.log(self.a1) -math.log(Mu)
        # print(" anpha ", anpha)
        # print(" beta ", beta)
        # print(" lamda ", lamda)
        delta = math.pow(beta,2)- 4*anpha*lamda
        if delta <0:
            return None
        elif delta==0:
            return -beta/(2*anpha)
        else:
            x1 = (-beta + math.sqrt(delta))/(2*anpha)
            x2 = (-beta - math.sqrt(delta)) / (2 * anpha)
            if x1 <= self.uth: #and x1>= self.lth :
                return x1
            else:
                return x2


    def getStandardFuzz(self,IFEObj, lamda=0.5):
        ins_obj= copy.copy(IFEObj)
        new_membership = ins_obj.getMembership()+lamda*ins_obj.getHesitance()
        new_nonmembership = ins_obj.getNonmembership() + (1-lamda)*ins_obj.getHesitance()
        ins_obj.setMembership(new_membership)
        ins_obj.setNonmembership(new_nonmembership)
        ins_obj.setHesitance(1-new_membership-new_nonmembership)
        return ins_obj


    def getStandardScriptation(self, Obj):
        new_obj = self.getStandardFuzz(Obj)
        scrip_value = self.getScriptation(new_obj)
        return scrip_value

    def getAdjustStandardFuzz(self,IFEObj, lamda=0.5):
        obj= copy.copy(IFEObj)
        new_membership = obj.getMembership() + \
                         (obj.getMembership()/(obj.getMembership()+obj.getNonmembership()))*obj.getHesitance()
        new_nonmembership = obj.getNonmembership() +\
                            (obj.getNonmembership() / (obj.getMembership()+obj.getNonmembership()))*obj.getHesitance()
        obj.setMembership(new_membership)
        obj.setNonmembership(new_nonmembership)
        obj.setHesitance(1-new_membership-new_nonmembership)
        return obj

    def getAdjustStandardScriptation(self, Obj):
        new_obj = self.getAdjustStandardFuzz(Obj)
        scrip_value = self.getScriptation(new_obj)
        return scrip_value

if __name__== "__main__":

    obj = IFE.IFE()
    obj.setMembership(0.9)
    obj.setNonmembership(0.0)
    obj.setHesitance(1-0.9)

    defuzzilier = IFE_defuzzilier(IFE_ins=obj)
    # obj.show()
    print("scrip value ", defuzzilier.getScriptation())
