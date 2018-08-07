#!/usr/bin/python3
"""K->pipi analysis code"""

import sys
import re
import h5py
import numpy as np
import read_file as rf
import math
from collections import defaultdict
from mpi4py import MPI
import h5jack

import kaondecompose # decompose the results, stored as 1d arrays, into multi-dimensional arrays
import kaonfileproc as kfp # process kaon files
import kaonpostproc as kpp # global container for results
import kaonmix # do mix subtraction
import kaonvac # vacuum subtraction

TSTEP12 = 8
LT_CHECK = 64
assert h5jack.LT == LT_CHECK, "Time extents do not match"
# the key structure is different, key doesn't end with @momentum string, e.g. @000
assert not h5jack.STILLSUB , "Vacuum subtraction not backwards compatible with this option"

# to do
# 1 mix3,mix4,sigma stuff
# bonus k->pi

# outline
# 1 read in diagrams

def filterOutSigma(unfiltered):
    """Filter out sigma"""
    return list(filter(lambda x: 'sigma' not in x, unfiltered))

def filterDiags(filter_str, diags_unfiltered):
    """Filter based on str"""
    return list(filter(lambda x: filter_str in x, diags_unfiltered))


def analyze():
    """Read in the k->x diagrams"""
    trajl = h5jack.trajlist()
    diags_unfiltered = h5jack.bublist()
    diags = {}
    purefilterlist = ['type1', 'type2', 'type3', 'type4']
    for filt in purefilterlist:
        diags[filt] = filterDiags(filt, diags_unfiltered)
    diags['mix4'] = filterDiags('mix4', diags['type4'])
    diags['mix3'] = filterDiags('mix3', diags['type3'])
    diags['mix3sigma'] = filterDiags('sigma', diags['mix3'])
    diags['mix3'] = filterOutSigma(diags['mix3'])
    diags['pipibubbles'] = filterDiags('Vdis', diags_unfiltered)
    diags['sigmabubbles'] = filterDiags('scalar-bubble', diags_unfiltered)
    diags['type2sigma'] = filterDiags('sigma' in diags['type2'])
    diags['type2'] = filterOutSigma(diags['type2'])
    diags['type3simga'] = filterDiags('sigma', diags['type3'])
    diags['type3'] = filterOutSigma(diags['type3'])

    sigmabubbles = h5jack.getbubbles(diags['sigmabubbles'])
    pipibubbles = h5jack.getbubbles(diags['pipibubbles'])

    # zeros the output
    for i in range(10):
        for keyirr in kpp.QOPI0[str(i)]: # max momentum is 2
            kpp.QOPI0[str(i)][keyirr] = np.zeros((len(trajl), LT_CHECK), dtype=np.complex)
            kpp.QOPI2[str(i)][keyirr] = np.zeros((len(trajl), LT_CHECK), dtype=np.complex)

    # add type 1,2,3 to the output
    kfp.proctype123(diags['type1'], trajl, 'type1')
    kfp.proctype123(diags['type2'], trajl, 'type2')
    kfp.proctype123(diags['type3'], trajl, 'type3')
    kfp.procSigmatype23(diags['type2sigma'], trajl, 'type2sigma')
    kfp.procSigmatype23(diags['type3sigma'], trajl, 'type3sigma')

    # form single jackknife blocks of the operators, which should be only projected types 1,2,3
    jackknifeOPS()

    # get the disconnected k->x pieces
    diags = get_kdiscon_fromfile(diags, trajl)

    # get mix coefficients from tK summed type4/2 and tK summed mix4
    alpha_kpipi = kaonmix.mixCoeffs(diags['type4_summed'], diags['mix4_summed'], trajl, 0)
    alpha_kpi = kaonmix.mixCoeffs(diags['type4_summed'], diags['mix4_summed'], trajl, 1) # useful for chipt study

    # do the mix4 vacuum subtraction, bubble composition, tK time averaging, jackknifing
    mix4to_pipi = kaonvac.vacSubtractMix4(diags['mix4_unsummed'], pipibubbles, trajl)
    mix4to_sigma = kaonvac.vacSubtractMix4(diags['mix4_unsummed'], sigmabubbles, trajl)

    # do vacuum subtraction, jackknife, project resulting type 4 onto operators
    kaonvac.vacSubtractType4(diags['type4_unsummed'], pipibubbles, trajl, 'pipi')
    kaonvac.vacSubtractType4(diags['type4_unsummed'], sigmabubbles, trajl, 'sigma')

    # do mix subtraction
    kaonmix.mixSubtract(alpha_ktopipi, diags['mix3'], mix4to_pipi, 'pipi')
    kaonmix.mixSubtract(alpha_ktopipi, diags['mix3'], mix4to_sigma, 'pipi')

    # write the results
    kpp.writeOut()

def get_kdiscon_fromfile(diags, trajl):
    """Get the mix4 and disconnected type4 diagrams from file"""
    # get type4 k->op piece (call it type4/2) from file
    diags['type4_unsummed'] = kfp.proctype4(diags['type4'], trajl, False)
    diags['type4_summed'] = kfp.proctype4(diags['type4'], trajl, True)

    # get mix diagrams from file
    diags['mix3'] = kfp.proctype123(diags['mix3'], trajl, 'mix3') # avg over Tk
    diags['mix3sigma'] = kfp.proctype123(diags['mix3sigma'], trajl, 'mix3') # avg over Tk
    diags['mix4_unsummed'] = kfp.procmix(diags['mix4'], trajl, 'mix4', False)
    diags['mix4_summed'] = kfp.procmix(diags['mix4'], trajl, 'mix4', True)
    return diags


def jackknifeOPS():
    """Jackknife operators."""
    OPS = [kpp.QOPI0, kpp.QOPI2, kpp.QOP_sigma]
    for opa in OPS:
        for i in np.arange(1, 11): # loop over operators
            for key in opa[str(i)]:
                if opa == kpp.QOPI0:
                    kpp.QOPI0[str(i)][key] = h5jack.dojackknife(opa[str(i)][key])
                elif opa == kpp.QOPI2:
                    kpp.QOPI2[str(i)][key] = h5jack.dojackknife(opa[str(i)][key])
                elif opa == kpp.QOP_sigma:
                    kpp.QOP_sigma[str(i)][key] = h5jack.dojackknife(opa[str(i)][key])
    jackknifeOPS.complete = True
jackknifeOPS.complete = False
            





# 2 decompose into pieces
# 3. do subtractions (vac, mix)
# 3.5 and jackknifing
# make into operators

#def ptokey(momdiag):
#    """Turns a momentum key into a momentum, then finds |p|"""
#    mom = rf.mom(momdiag)
#    key = np.dot(mom, mom)
#return key







def main():
    """do program"""


if __name__ == '__main__':
    main()
