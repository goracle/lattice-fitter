#!/usr/bin/python3
"""K->pipi analysis code"""

from collections import defaultdict
import numpy as np
from latfit.utilities.postprod import checkblks, bubblks, h5jack

# import kaondecompose # decompose the results,
# stored as 1d arrays, into multi-dimensional arrays
import latfit.utilities.kaonanalysis.kaonfileproc as kfp # process kaon files
import latfit.utilities.kaonanalysis.kaonpostproc as kpp # global container for results
import latfit.utilities.kaonanalysis.kaonmix as kaonmix # do mix subtraction
import latfit.utilities.kaonanalysis.kaonvac as kaonvac # vacuum subtraction

TSTEP12 = 2
kfp.TSTEP12 = TSTEP12
LT_CHECK = 4
if checkblks.FREEFIELD:
    LT_CHECK = 32
assert h5jack.LT == LT_CHECK, "Time extents do not match"
# the key structure is different,
# key doesn't end with @momentum string, e.g. @000
assert not bubblks.STILLSUB,\
    "Vacuum subtraction not backwards compatible with this option"

# to do
# 1 mix3,mix4,sigma stuff
# bonus k->pi

# outline
# 1 read in diagrams

def filter_out_sigma(unfiltered):
    """Filter out sigma"""
    ret = list(filter(lambda x: 'sigma' not in x, unfiltered))
    assert ret, "filter out sigma returned empty list"
    return ret

def filter_out_mix(unfiltered):
    """Filter out sigma"""
    ret = list(filter(lambda x: 'mix' not in x, unfiltered))
    assert ret, "filter out mix returned empty list"
    return ret

def filter_diags(filter_str, diags_unfiltered):
    """Filter based on str"""
    ret = list(filter(lambda x: filter_str in x, diags_unfiltered))
    assert ret, "Filter diags returned empty list for filter str="+str(
        filter_str)
    return ret

def filter_out_ktopi(unfiltered):
    """Filter out the k->pi digrams because we are not doing that study yet
    """
    ret = list(filter(lambda x: 'ktopi' not in x, unfiltered))
    assert ret, "filter out k->pi returned empty list"
    return ret

def analyze():
    """Read in the k->x diagrams"""
    trajl = h5jack.trajlist()
    diags_unfiltered = bubblks.bublist()
    diags_unfiltered = filter_out_ktopi(diags_unfiltered)
    diags = {}
    purefilterlist = ['type1', 'type2', 'type3', 'type4']
    for filt in purefilterlist:
        diags[filt] = filter_diags(filt, diags_unfiltered)
        assert diags[filt], str(filt)+" diagrams missing from base set."

    # mix3,4
    diags['mix4'] = filter_diags('mix4', diags['type4'])
    diags['type4'] = filter_out_mix(diags['type4'])
    diags['mix3'] = filter_diags('mix3', diags['type3'])
    diags['type3'] = filter_out_mix(diags['type3'])

    # sigma specific diagram lists
    diags['mix3sigma'] = filter_diags('sigma', diags['mix3'])
    diags['mix3'] = filter_out_sigma(diags['mix3'])
    diags['type2sigma'] = filter_diags('sigma', diags['type2'])
    diags['type2'] = filter_out_sigma(diags['type2'])
    diags['type3sigma'] = filter_diags('sigma', diags['type3'])
    diags['type3'] = filter_out_sigma(diags['type3'])

    # get bubbles
    diags['pipibubbles'] = filter_diags('Vdis', diags_unfiltered)
    diags['sigmabubbles'] = filter_diags('scalar-bubble', diags_unfiltered)
    sigmabubbles = bubblks.getbubbles(diags['sigmabubbles'], trajl)
    pipibubbles = bubblks.getbubbles(diags['pipibubbles'], trajl)

    # zeros the output to be safe
    for i in np.arange(1, 11):

        kpp.QOPI0[str(i)] = defaultdict(lambda: np.zeros(
            (len(trajl), LT_CHECK), dtype=np.complex))

        kpp.QOPI2[str(i)] = defaultdict(lambda: np.zeros(
            (len(trajl), LT_CHECK), dtype=np.complex))

        kpp.QOP_SIGMA[str(i)] = defaultdict(lambda: np.zeros(
            (len(trajl), LT_CHECK), dtype=np.complex))

    # add type 1,2,3 to the output
    kfp.proctype123(diags['type1'], trajl, 'type1')
    kfp.proctype123(diags['type2'], trajl, 'type2')
    kfp.proctype123(diags['type3'], trajl, 'type3')
    kfp.proc_sigma_type23(diags['type2sigma'], trajl, 'type2')
    kfp.proc_sigma_type23(diags['type3sigma'], trajl, 'type3')

    # form single jackknife blocks of the operators, which should be only projected types 1,2,3
    print("jackknifing ops")
    jackknife_ops()

    # get the disconnected k->x pieces
    print("getting disconnected k")
    diags = get_kdiscon_fromfile(diags, trajl)

    # get mix coefficients from tK summed type4/2 and tK summed mix4
    print("getting mix coefficients")
    alpha_kpipi = kaonmix.mix_coeffs(diags['type4_summed'],
                                     diags['mix4_summed'], trajl, 0)
    # useful for chipt study
    #alpha_kpi = kaonmix.mix_coeffs(diags['type4_summed'],
    #                              diags['mix4_summed'], trajl, 1)

    # do the mix4 vacuum subtraction, bubble composition,
    # tK time averaging, jackknifing
    print("vacuum subtracting mix 4")
    mix4to_pipi = kaonvac.vac_subtract_mix4(diags['mix4_unsummed'],
                                            pipibubbles, trajl)
    mix4to_sigma = kaonvac.vac_subtract_mix4(diags['mix4_unsummed'],
                                             sigmabubbles, trajl)

    # do vacuum subtraction, jackknife,
    # project resulting type 4 onto operators
    print("vacuum subtracting type 4")
    kaonvac.vac_subtract_type4(diags['type4_unsummed'],
                               pipibubbles, trajl, 'pipi')
    kaonvac.vac_subtract_type4(diags['type4_unsummed'],
                               sigmabubbles, trajl, 'sigma')

    # do mix subtraction
    assert jackknife_ops.complete, "Operators need to be jackknifed"+\
        " before mix subtraction."
    print("performing mix subtraction")
    kaonmix.mix_subtract(alpha_kpipi, diags['mix3'], mix4to_pipi, 'pipi', len(trajl))
    kaonmix.mix_subtract(alpha_kpipi, diags['mix3'], mix4to_sigma, 'pipi', len(trajl))

    # write the results
    print("writing kaon output")
    kpp.write_out()

def get_kdiscon_fromfile(diags, trajl):
    """Get the mix4 and disconnected type4 diagrams from file"""
    # get type4 k->op piece (call it type4/2) from file
    diags['type4_unsummed'] = kfp.proctype4(diags['type4'], trajl, False)
    diags['type4_summed'] = kfp.proctype4(diags['type4'], trajl, True)

    # get mix diagrams from file
    diags['mix3'] = kfp.proctype123(diags['mix3'], trajl, 'mix3') # avg over Tk
    diags['mix3sigma'] = kfp.proctype123(diags['mix3sigma'], trajl, 'mix3') # avg over Tk
    diags['mix4_unsummed'] = kfp.procmix4(diags['mix4'], trajl, False)
    diags['mix4_summed'] = kfp.procmix4(diags['mix4'], trajl, True)
    return diags


def jackknife_ops():
    """Jackknife operators."""
    for i in np.arange(1, 11): # loop over operators
        # keylist = []
        for key in kpp.QOPI0[str(i)]:
            kpp.QOPI0[str(i)][key] = h5jack.dojackknife(
                kpp.QOPI0[str(i)][key])
            kpp.QOPI2[str(i)][key] = h5jack.dojackknife(
                kpp.QOPI2[str(i)][key])
            kpp.QOP_SIGMA[str(i)][key] = h5jack.dojackknife(
                kpp.QOP_SIGMA[str(i)][key])
    jackknife_ops.complete = True
jackknife_ops.complete = False



# 2 decompose into pieces
# 3. do subtractions (vac, mix)
# 3.5 and jackknifing
# make into operators

# def ptokey(momdiag):
#     """Turns a momentum key into a momentum, then finds |p|"""
#     mom = rf.mom(momdiag)
#     key = kdot(mom, mom)
# return key







def main():
    """do program"""
    analyze()


if __name__ == '__main__':
    main()
