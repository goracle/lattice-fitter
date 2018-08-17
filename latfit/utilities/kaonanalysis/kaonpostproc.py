"""Storage for post processed operators.  After these are filled we write to disk"""
import numpy as np
import h5py

QOPI0 = {} # operator dictionary
QOPI2 = {} # operator dictionary, I2
QOP_sigma = {} # sigma operator dictionary

for i in np.arange(1, 11):
    QOPI0[str(i)] = {} # momentum dictionary
    QOPI2[str(i)] = {} # momentum dictionary, I2
    QOP_sigma[str(i)] = {} # momentum dictionary


def writeOut():
    """Write the result to file"""
    keyarr = []
    for i in np.arange(1, 11):
        for keyirr in QOPI0[str(i)]:
            momrel, kpitsep = keyirr.split('@')
            keyarr.append(keyirr)
            assert keyirr in QOPI0[str(i)], "tsep = "+kpitsep+\
                " not in Q_"+str(i)+", I=0"
            assert keyirr in QOPI2[str(i)], "tsep = "+kpitsep+\
                " not in Q_"+str(i)+", I=2"
    for key in keyarr:
        momrel, kpitsep = key.split('@')
        momrel = int(momrel)
        if momrel == 0:
            opstr = 'S_pipi'
        elif momrel == 1:
            opstr = 'pipi'
        elif momrel == 2:
            opstr = 'UUpipi'
        elif momrel == 3:
            opstr = 'U2pipi'
        else:
            print("bad momentum in key=", key)
            raise
        for i in np.arange(1, 11):
            filestr = 'Q'+str(i)+'_deltat_'+str(kpitsep)
            filestr = filestr+'_'+opstr
            filestr_sigma = filestr+'_sigma'
            fn1 = h5py.File('I0/'+filestr, 'w')
            fn1[filestr] = QOPI0[str(i)][key]
            fn1.close()
            fn1 = h5py.File('I2/'+filestr, 'w')
            fn1[filestr] = QOPI2[str(i)][key]
            fn1.close()
            if momrel == 0:
                fn1 = h5py.File('I0/'+filestr_sigma, 'w')
                fn1[filestr] = QOP_sigma[str(i)][key]
                fn1.close()
