#!/usr/bin/python3
"""Make the ratio

C_pipi(t)-C_pipi(t+1)/
(
C_pi^2(t)-C_pi^2(t+1)
)
where C_pi is the single pion correlator with the same center of mass

"""
import sys
import re
import copy
import glob
import numpy as np
import h5py
import read_file as rf
from sum_blks import isoproj
from h5jack import TSEP, LT, overall_coeffs, h5sum_blks, avg_irreps
import op_compose as opc

MOM = [0, 0, 0]
STYPE = 'hdf5'


def main():
    """Make the ratio

    C_pipi(t)-C_pipi(t+1)/
    (
    C_pi^2(t)-C_pi^2(t+1)
    )
    where C_pi is the single pion correlator with the same center of mass
    usage:
    <pipi correlator (hdf5 file)> <center of mass momentum; magnitude,
    or triple specifying specific 3 momentum (e.g. 0 0 1)>
    """
    if len(sys.argv) == 2:
        ptot = MOM
    elif len(sys.argv) == 3:
        ptotstr = sys.argv[2]+'unit'+('s' if int(sys.argv[2]) != 1 else '')
    elif len(sys.argv) in [5, 6]:
        ptot = np.array([sys.argv[i+2] for i in range(3)])
        dosub = True if len(sys.argv) == 5 else False
    else:
        print("Number of arguments error."+\
              "  Needs input file and COM momentum")
        print("e.g.: test.dat 0 0 1")
        raise
    if len(sys.argv) != 3:
        ptotstr = rf.ptostr(ptot)
    for i in (True, False):
        do_ratio(ptotstr, i)

def piondirect():
    """Do pion ratio unsummed."""
    base = 'pioncorrChk_*_unsummed.jkdat'
    baseglob = glob.glob(base)
    allblks = {}
    count = {}
    numt = None
    for num1, fname in enumerate(baseglob):
        fn1 = h5py.File(fname, 'r')
        momf = np.asarray(rf.mom(fname))
        numt = len(momf) if numt is None else numt
        assert numt == len(momf), "inconsistent config number"
        for i in fn1:
            toppi = np.array(fn1[i])
            save1 = i
        for num2, gname in enumerate(baseglob):
            momg = np.asarray(rf.mom(gname))
            assert numt == len(momg), "inconsistent config number"
            gn1 = h5py.File(gname, 'r')
            for i in gn1:
                bottompi = np.array(gn1[i])
                save2 = i
            shiftpi = np.roll(bottompi, TSEP, axis=1)

            # compute the first topology
            top1 = shiftpi*toppi
            top1 = np.roll(top1, -1*TSEP, axis=2)
            top1 = np.mean(top1, axis=1)
            assert top1[0][0] > 100

            # compute the second topology
            top2 = np.roll(shiftpi, -2*TSEP, axis=2)*toppi
            top2 = np.mean(top2, axis=1)

            # for use in I=1
            top3 = -1*top2

            assert top2[0][0] > 100
            momstr = 'sep'+str(TSEP)+'_mom1src'+rf.ptostr(momf)+\
                '_mom2src'+rf.ptostr(momg)+'_mom1snk'
            key1 = addfigd(momstr)+rf.ptostr(momf)
            key3halves = addfigdvec(momstr)+rf.ptostr(momf)
            key2 = addfigd(momstr)+rf.ptostr(momg)
            key3 = addfigdvec(momstr)+rf.ptostr(momg)
            if 'mom1src000_mom2src001_mom1snk001' in key1:
                pass
                #print(fname, gname)
            if 'mom1src000_mom2src001_mom1snk001' in key2:
                pass
                #print(fname, gname, '2')
            if 'mom1src000_mom2src001_mom1snk001' in key1 or\
               'mom1src000_mom2src001_mom1snk001' in key2:
                #print(save1)
                #print(save2)
                temp = np.mean(top2,axis=0)
                temp = np.real(temp)
                temp2 = np.mean(toppi, axis=0)
                temp2 = np.mean(temp2, axis=0)
                temp2= np.real(temp2)
                #print(np.log(temp[5]/temp[6]))
                #print(np.log(temp2[5]/temp2[6]))
                #sys.exit(0)
            assert toppi.shape == bottompi.shape, "shape mismatch"
            if key1 not in allblks:
                allblks[key1] = np.zeros((len(toppi), LT), dtype=np.complex)
                count[key1] = 1
            else:
                count[key1] += 1
            if key3halves not in allblks:
                allblks[key3halves] = np.zeros((len(toppi), LT), dtype=np.complex)
                count[key3halves] = 1
            else:
                count[key3halves] += 1
            if key2 not in allblks:
                allblks[key2] = np.zeros((len(toppi), LT), dtype=np.complex)
                count[key2] = 1
            else:
                count[key2] += 1
            if key3 not in allblks:
                allblks[key3] = np.zeros((len(toppi), LT), dtype=np.complex)
                count[key3] = 1
            else:
                count[key3] += 1
            allblks[key1] += top1/2
            allblks[key3halves] += top1/2
            allblks[key2] += top2/2
            allblks[key3] += top3/2

    for i in allblks:
        try:
            assert count[i] == 2
        except AssertionError:
            assert count[i] == 1
            print("topologies used:", count[i])
            print(i)
    temp = opc.op_list(stype=STYPE)
    ocs = overall_coeffs(
        isoproj(False, 0, dlist=list(
            allblks.keys()), stype=STYPE), opc.op_list(stype=STYPE))
    assert isinstance(ocs, dict)
    for i in list(ocs):
        ocs[i+'_pisq'] = ocs[i]
        del ocs[i]
    h5sum_blks(allblks, ocs, (numt, LT))
    avg_irreps('_pisq.jkdat')

def coeffto1(ocs):
    """Set coefficients to 1"""
    for opa in ocs:
        for i in range(len(ocs[opa])):
            base, coeff = ocs[opa][i]
            ocs[opa][i] = (base, 1.0)
    return ocs

def aroundworldsubtraction(allblks, deltat=3):
    """Around the world subtraction (by 3)"""
    for blk in allblks:
        t1blk = copy.deepcopy(np.asarray(allblks[blk]))
        t2blk = copy.deepcopy(np.roll(allblks[blk], deltat, axis=1))
        retblk = t1blk-t2blk
        assert retblk[0][8] < 0
        assert retblk[0][9] < 0
        assert retblk[0][10] < 0
        assert retblk[0][11] < 0
        assert retblk[0][11] < 0
        assert abs(retblk[0][11]) < abs(retblk[0][8])
        allblks[blk] = copy.deepcopy(retblk)
    return allblks

def directratio(allblks, deltat_matrix=3):
    """Take ratio of two directo diagrams
    (1x1 GEVP)
    """
    for blk in allblks:
        temp = copy.deepcopy(np.array(allblks[blk]))
        temp2 = copy.deepcopy(np.array(allblks[blk]))
        assert abs(temp[0][11]) < abs(temp[0][8])
        temp3 = temp[0][8]
        temp = np.roll(temp, deltat_matrix, axis=1)
        assert temp[0][11] == temp3
        assert abs(temp[0][13]) > abs(allblks[blk][0][13])
        assert abs(temp[0][13]) > abs(temp2[0][13])
        allblks[blk] = temp2/temp
        assert abs(allblks[blk][0][13]) < 1
    return allblks

def addfigdvec(strin):
    """Add figure D vec to string"""
    return 'FigureD_vec_'+strin
 

def addfigd(strin):
    """Add figure D to string"""
    return 'FigureD_'+strin
            


def do_ratio(ptotstr, dosub):
    """Make the ratio for a given ptotal string and 
    bool for whether or not to do subtraction in the numerator
    """
    filename = sys.argv[1]+'.jkdat' if len(sys.argv[1].split('.')) == 1 else sys.argv[1]
    fn1 = h5py.File(filename, 'r')
    data, group = anticipate(fn1)
    pionstr = 'pioncorr_mom'+ptotstr
    pionstr = 'pioncorrChk_mom'+ptotstr
    print("using pion correlator:", pionstr)
    gn1 = h5py.File(pionstr+'.jkdat', 'r')
    pion = np.array(gn1[pionstr])
    multsub = 1.0 if dosub else 0.0
    C_pipiminus = np.array(
        [
            [
                data[config][time]-data[config][time+1]*multsub
                for time in range(len(data[0])-1)
            ]
            for config in range(len(data))
        ]
    )
    pionminus = np.array(
        [
            [
                (pion[config][time])**2-(pion[config][time+1])**2
                for time in range(len(data[0])-1)
            ]
            for config in range(len(data))
        ]
    )
    filesplit = filename.split('.')
    ext = '.jkdat'
    suffix = ext if dosub else '_nosub'+ext
    hn1 = h5py.File(filesplit[0]+'_pionratio'+suffix, 'w')
    hn1[filesplit[0]+'_pionratio'] = C_pipiminus/pionminus
    hn1.close()
    print("Finished with", sys.argv[1])
    return


def anticipate(fn):
    """Anticipate the hdf5 structure: one dataset in one group"""
    group = ''
    for i in fn:
        group = i
    data = ''
    for i in fn[group]:
        data = i
    return np.array(fn[group+'/'+i]), group

        

if __name__ == '__main__':
    piondirect()
