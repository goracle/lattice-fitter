"""Vacuum subtract disconnected k->x diagrams."""

import import kaonfileproc as kfp
import kaonpostproc as kpp
import h5jack
import kaonprojop
from kaonanalysis import LT_CHECK

def vacSubtractMix4(mix4dict, sinkbubbles, trajl):
    """Vacuum subtract type4"""

    sinksub = bubsub(sinkbubbles)
    ltraj = len(trajl)

    # jackknife type 4

    aftersub = {}
    for time in range(LT_CHECK):
        aftersub[subkey] = np.zeros((ltraj, 8, 4, LT_CHECK), dtype=np.complex)
    for momdiag in mix4dict:
        aftersub[momdiag] = np.zeros((ltraj, 2, LT_CHECK), dtype=np.complex)
        for fidx in range(2): # loop over gamma structure in the mix diagram (g5, unit)
            for tdis in range(LT_CHECK):

                # get src bubbles
                momdiagc = momdiag+'@000' # for backwards compatibility, means key@ptotal, ptotal=000 since Kaon is at rest
                temp_dict = {}
                temp_dict[momdiagc] = type4dict[mom_diag][:, fidx, tdis, :]
                srcsub = h5jack.bubsub(temp_dict)

                # dict of averaged bubbles, to subtract
                subdict = {**srcsub, **sinksub}

                # dict of uncomposed bubbles
                bubbles = {**temp_dict, **sinkbubbles}

                # do the vac subtraction, avg over tk
                bubblk = h5jack.dobubjack(bubbles, subdict)[momdiag]
                for tsep_kpi in range(LT_CHECK):
                    subkey = momdiag+"_deltat_"+str(tsep_kpi)
                    aftersub[subkey][:, fidx, tdis] = bubblk[:, tsep_kpi]

    return aftersub
        


def vacSubtractType4(type4dict, sinkbubbles, trajl, otype):
    """Vacuum subtract type4"""

    # for reference
    # shapeType4 = (ltraj, 8, 4, LT_CHECK, LT_CHECK)
    # shapeMix4 = (ltraj, 2, LT_CHECK, LT_CHECK)

    # to do, loop over tsep_kpi

    sinksub = bubsub(sinkbubbles)

    aftersub = {}
    for time in range(LT_CHECK):
        aftersub[subkey] = np.zeros((ltraj, 8, 4, LT_CHECK), dtype=np.complex)
    for momdiag in type4dict:
        for conidx in range(8):
            for gcombidx in range(4):
                for tdis in range(LT_CHECK):

                    temp_dict = {}
                    momdiagc = momdiag+'@000' # for backwards compatibility, means key@ptotal, ptotal=000 since Kaon is at rest
                    temp_dict[momdiagc] = type4dict[momdiag][:, conidx, gcombidx, tdis, :]
                    srcsub = h5jack.bubsub(temp_dict)

                    # dict of averaged bubbles, to subtract
                    subdict = {**srcsub, **sinksub}

                    # dict of uncomposed bubbles
                    bubbles = {**temp_dict, **sinkbubbles}

                    # do the vac subtraction, avg over tk
                    bubblk = dobubjack(bubbles, subdict)[momdiag]

                    # now, use the result to create type4 diagrams with defined tsep_kpi
                    for tsep_kpi in range(LT_CHECK):
                        subkey = momdiag+"_deltat_"+str(tsep_kpi)
                        aftersub[subkey][:, conidx, gcombidx, tdis] = bubblk[:, tsep_kpi]

    # project finally onto the operators

    for num, traj in enumerate(trajl):
        for momdiag in aftersub:
            keyirr = kfp.genKey(momdiag)
            for i in np.arange(1, 11):
                if otype == 'pipi':
                    kpp.QOPI0[str(i)][keyirr][num] += kaonprojop.QiprojType4(aftersub[momdiag][num], i, 'I0')
                elif otype == 'sigma':
                    kpp.QOP_sigma[str(i)][keyirr][num] += kaonprojop.QiprojSigmaType4(aftersub[momdiag][num], i, 'I0')
                else:
                    assert None, "bad otype"
