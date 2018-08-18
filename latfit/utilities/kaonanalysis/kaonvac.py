"""Vacuum subtract disconnected k->x diagrams."""

import sys
from collections import defaultdict
import kaonfileproc as kfp
import kaonpostproc as kpp
import latfit.utilities.h5jack
from latfit.utilities.h5jack import LT as LT_CHECK
import kaonprojop
import numpy as np

def vacSubtractMix4(mix4, sinkbubbles, trajl, otype):
    """Vacuum subtract type4"""

    sinksub = latfit.utilities.h5jack.bubsub(sinkbubbles)
    ltraj = len(trajl)

    # jackknife type 4

    aftersub = {}
    # for time in range(LT_CHECK):
    # aftersub[subkey] = np.zeros((ltraj, 2, LT_CHECK), dtype=np.complex)
    aftersub = defaultdict(lambda: np.zeros((ltraj, 2, LT_CHECK),
                                            dtype=np.complex))
    for fidx in range(2): # loop over gamma structure in the mix diagram (g5, unit)
        for tdis in range(LT_CHECK):

            # get src bubbles

            ## for backwards compatibility,
            ## means key@ptotal, ptotal=000 since Kaon is at rest
            momdiagc = 'type4@000' # both vac subtractions are the same
            temp_dict = {}
            temp_dict[momdiagc] = mix4[:, fidx, tdis, :]
            srcsub = latfit.utilities.h5jack.bubsub(temp_dict)

            # dict of averaged bubbles, to subtract
            subdict = {**srcsub, **sinksub}

            # dict of uncomposed bubbles
            bubbles = {**temp_dict, **sinkbubbles}

            print("vac: fidx, tdis =", fidx, tdis)
            # do the vac subtraction, avg over tk
            bubblks = latfit.utilities.h5jack.dobubjack(
                bubbles, subdict, skipVBub2=True)

            for blkname in bubblks:
                for tsep_kpi in range(LT_CHECK):
                    subkey = blkname+"_mix4_deltat_"+str(tsep_kpi)
                    aftersub[subkey][:, fidx, tdis] = bubblks[
                        blkname][:, tsep_kpi]

    return aftersub
        


def vacSubtractType4(type4, sinkbubbles, trajl, otype):
    """Vacuum subtract type4"""

    # for reference
    # shapeType4 = (ltraj, 8, 4, LT_CHECK, LT_CHECK)
    # shapeMix4 = (ltraj, 2, LT_CHECK, LT_CHECK)

    # to do, loop over tsep_kpi

    sinksub = latfit.utilities.h5jack.bubsub(sinkbubbles)

    aftersub = {}
    for conidx in range(8):
        for gcombidx in range(4):
            for tdis in range(LT_CHECK):

                temp_dict = {}
                # for backwards compatibility,
                # means key@ptotal, ptotal=000 since Kaon is at rest
                momdiagc = 'type4@000'
                temp_dict[momdiagc] = type4[:, conidx, gcombidx, tdis, :]
                srcsub = latfit.utilities.h5jack.bubsub(temp_dict)

                # dict of averaged bubbles, to subtract
                subdict = {**srcsub, **sinksub}

                # dict of uncomposed bubbles
                bubbles = {**temp_dict, **sinkbubbles}

                # do the vac subtraction, avg over tk
                print("vac: conidx, gcombidx, tdis =",
                      conidx, gcombidx, tdis)
                bubblks = latfit.utilities.h5jack.dobubjack(
                    bubbles, subdict, skipVBub2=True)

                # now, use the result to create type4 diagrams,
                # with defined tsep_kpi
                for blkname in bubblks:
                    for tsep_kpi in range(LT_CHECK):
                        subkey = blkname+"_deltat_"+str(tsep_kpi)
                        aftersub[subkey] = np.zeros(
                            (len(trajl), 8, 4, LT_CHECK), dtype=np.complex)
                        aftersub[subkey][
                            :, conidx, gcombidx, tdis] = bubblks[
                                blkname][:, tsep_kpi]

    # project finally onto the operators

    for num, traj in enumerate(trajl):
        for momdiag in aftersub:
            keyirr = kfp.genKey(momdiag)
            for i in np.arange(1, 11):
                if otype == 'pipi':
                    assert str(i) in kpp.QOPI0, "Missing Q:"+str(i)
                    kpp.QOPI0[str(i)][
                        keyirr][num] += kaonprojop.QiprojType4(
                            aftersub[momdiag][num], i)
                elif otype == 'sigma':
                    assert str(i) in kpp.QOP_sigma, "Missing Q:"+str(i)
                    kpp.QOP_sigma[str(i)][keyirr][
                        num] += kaonprojop.QiprojSigmaType4(
                            aftersub[momdiag][num], i)
                else:
                    assert None, "bad otype"
