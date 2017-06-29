import numpy as np

def inverse_jk(REUSE,time_range,num_configs=-1):
    #REUSE[config][time]
    if num_configs<0:
        num=len(REUSE)-1
    else:
        num=num_configs-1
    REUSE_INV=np.zeros((num_configs,len(time_range)))
    REUSE_INV=np.sum(REUSE,0)-num*REUSE
    return REUSE_INV
