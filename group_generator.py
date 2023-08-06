''' giap
    generate groups profiles by a random sampling mechanism
'''
import pandas as pd
from random import sample
def groupsGenerator(U,Gsize,N):
    # U is a list of user_ID
    #tao danh muc nhom bang random sampling mechanism
    #store User_ID (not User's position index)
    gp=[]
    for i in range(N):
        sample_i=sample(U,Gsize)
        # print('group i',sample_i)
        gp.append(sample_i)
    g_profiles= pd.DataFrame(gp)
    return g_profiles


