"""Storage for post processed operators.  After these are filled we write to disk"""
import os
import numpy as np
import h5py

QOPI0 = {} # operator dictionary
QOPI2 = {} # operator dictionary, I2
QOP_SIGMA = {} # sigma operator dictionary

for j in np.arange(1, 11):
    QOPI0[str(j)] = {} # momentum dictionary
    QOPI2[str(j)] = {} # momentum dictionary, I2
    QOP_SIGMA[str(j)] = {} # momentum dictionary

def querykeys(qop, descr):
    """For debug purposes.  Print keys in qop with description descr"""
    print("printing keys in", descr)
    for i in qop:
        print(i)
        for j in qop[str(i)]:
            print(j)
    print("done printing keys in", descr)

def correct_dagger_factor():
    """Need an overall -1 due to dagger not being taken on K
    """
    for i in np.arange(1,11):
        for keyirr in QOPI2[str(i)]:
            QOPI2[str(i)][keyirr] *= -1
        for keyirr in QOPI0[str(i)]:
            QOPI0[str(i)][keyirr] *= -1


def check_consistency_keys(keyarr):
    """Check key consistency"""
    for i in np.arange(1, 11):
        print("printing I2 keys")
        for keyirr in QOPI2[str(i)]:
            print(keyirr)
        print("printing I0 keys")
        for keyirr in QOPI0[str(i)]:
            print(keyirr)
        print("checking that I=0 has at least same keys as I=2")
        for keyirr in QOPI2[str(i)]:
            print(keyirr)
            _, kpitsep = keyirr.split('@')
            keyarr.add(keyirr)
            assert keyirr in QOPI0[str(i)], "tsep = "+kpitsep+\
                " not in Q_"+str(i)+", I=0"
            assert keyirr in QOP_SIGMA[str(i)], "tsep = "+kpitsep+\
                " not in Q_"+str(i)+", sigma"
    return keyarr



def write_out():
    """Write the result to file"""
    correct_dagger_factor()
    keyarr = set()
    if not os.path.isdir('I0'):
        os.makedirs('I0')
    if not os.path.isdir('I2'):
        os.makedirs('I2')
    keyarr = check_consistency_keys(keyarr)
    keyarr = sorted(list(keyarr))
    for key in keyarr:
        momrel, kpitsep = key.split('@')
        momrel = int(momrel)
        opstr = getopstr(momrel, key)
        for i in np.arange(1, 11):
            filestr0 = 'Q'+str(i)+'_I0_deltat_'+str(kpitsep)
            filestr2 = 'Q'+str(i)+'_I2_deltat_'+str(kpitsep)
            filestr_pipi0 = filestr0+'_'+opstr
            filestr_pipi2 = filestr2+'_'+opstr
            if not os.path.isfile("I0/"+filestr_pipi0):
                print("writing file: ", "I0/"+filestr_pipi0)
                fn1 = h5py.File('I0/'+filestr_pipi0, 'w')
                fn1[filestr_pipi0] = QOPI0[str(i)][key]
                fn1.close()
            else:
                print("skipping extant file: ", "I0/"+filestr_pipi0)
            if not os.path.isfile("I2/"+filestr_pipi2):
                print("writing file: ", "I2/"+filestr_pipi2)
                fn1 = h5py.File('I2/'+filestr_pipi2, 'w')
                fn1[filestr_pipi2] = QOPI2[str(i)][key]
                fn1.close()
            else:
                print("skipping extant file: ", "I2/"+filestr_pipi2)
            if momrel == 0:
                filestr_sigma = filestr0+'_sigma'
                if not os.path.isfile("I0/"+filestr_sigma):
                    print("writing file: ", "I0/"+filestr_sigma)
                    gn1 = h5py.File('I0/'+filestr_sigma, 'w')
                    gn1[filestr_sigma] = QOP_SIGMA[str(i)][key]
                    gn1.close()
                else:
                    print("skipping extant file: ", "I0/"+filestr_sigma)

def getopstr(momrel, key):
    """Get the operator string based on the relative momentum between pions"""
    assert momrel in [0, 1, 2, 3], "bad momentum in key="+str(key)
    if momrel == 0:
        opstr = 'S_pipi'
    elif momrel == 1:
        opstr = 'pipi'
    elif momrel == 2:
        opstr = 'UUpipi'
    elif momrel == 3:
        opstr = 'U2pipi'
    return opstr
