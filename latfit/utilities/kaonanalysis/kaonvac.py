"""Vacuum subtract disconnected k->x diagrams."""

from collections import defaultdict
import numpy as np
import latfit.utilities.kaonanalysis.kaonfileproc as kfp
import latfit.utilities.kaonanalysis.kaonpostproc as kpp
import latfit.utilities.postprod.h5jack
from latfit.utilities.postprod.h5jack import LT as LT_CHECK
import latfit.utilities.kaonanalysis.kaonprojop as kaonprojop

def vac_subtract_mix4(mix4, sinkbubbles, trajl):
    """Vacuum subtract type4"""

    sinksub = latfit.utilities.postprod.bubblks.bubsub(sinkbubbles)

    # jackknife type 4

    aftersub = {}
    # for time in range(LT_CHECK):
    # aftersub[subkey] = np.zeros((ltraj, 2, LT_CHECK), dtype=np.complex)
    aftersub = defaultdict(lambda: np.zeros((len(trajl), 2, LT_CHECK),
                                            dtype=np.complex))
    # loop over gamma structure in the mix diagram (g5, unit)
    for fidx in range(2):
        for tdis in range(LT_CHECK):

            # get src bubbles

            ## for backwards compatibility,
            ## means key@ptotal, ptotal=000 since Kaon is at rest
            momdiagc = 'type4@000' # both vac subtractions are the same
            temp_dict = {}
            temp_dict[momdiagc] = mix4[:, fidx, tdis, :]
            srcsub = latfit.utilities.postprod.bubblks.bubsub(temp_dict)

            # dict of averaged bubbles, to subtract
            subdict = {**srcsub, **sinksub}

            # dict of uncomposed bubbles
            bubbles = {**temp_dict, **sinkbubbles}

            print("vac: fidx, tdis =", fidx, tdis)
            # do the vac subtraction, avg over tk
            bubblks = latfit.utilities.postprod.bubblks.dobubjack(
                bubbles, subdict, skip_v_bub2=True)

            for blkname in bubblks:
                for tsep_kpi in range(LT_CHECK):
                    aftersub[
                        blkname+"_mix4_deltat_"+str(tsep_kpi)][
                            :, fidx, tdis] = bubblks[
                                blkname][:, tsep_kpi]

    return aftersub


def vac_subtract_type4(type4, sinkbubbles, trajl, otype):
    """Vacuum subtract type4"""

    # for reference
    # shape_type4 = (ltraj, 8, 4, LT_CHECK, LT_CHECK)
    # shape_mix4 = (ltraj, 2, LT_CHECK, LT_CHECK)

    # to do, loop over tsep_kpi

    sinksub = latfit.utilities.postprod.bubblks.bubsub(sinkbubbles)

    aftersub = {}
    for conidx in range(8):
        for gcombidx in range(4):
            for tdis in range(LT_CHECK):

                temp_dict = {}
                # for backwards compatibility,
                # means key@ptotal, ptotal=000 since Kaon is at rest
                temp_dict['type4@000'] = type4[:, conidx, gcombidx, tdis, :]
                srcsub = latfit.utilities.postprod.bubblks.bubsub(temp_dict)

                # dict of averaged bubbles, to subtract
                subdict = {**srcsub, **sinksub}

                # dict of uncomposed bubbles
                bubbles = {**temp_dict, **sinkbubbles}

                # do the vac subtraction, avg over tk
                print("vac: conidx, gcombidx, tdis =",
                      conidx, gcombidx, tdis)
                bubblks = latfit.utilities.postprod.bubblks.dobubjack(
                    bubbles, subdict, skip_v_bub2=True)

                # now, use the result to create type4 diagrams,
                # with defined tsep_kpi
                aftersub = fill_aftersub(aftersub, bubblks, trajl,
                                         (conidx, gcombidx, tdis))

    # project finally onto the operators
    final_projection_vactype4(trajl, aftersub, otype)

def fill_aftersub(aftersub, bubblks, trajl, indices):
    """Fill the after sub dict
    """
    conidx, gcombidx, tdis = indices
    for blkname in bubblks:
        for tsep_kpi in range(LT_CHECK):
            subkey = blkname+"_deltat_"+str(tsep_kpi)
            aftersub[subkey] = np.zeros(
                (len(trajl), 8, 4, LT_CHECK), dtype=np.complex)
            aftersub[subkey][
                :, conidx, gcombidx, tdis] = bubblks[
                    blkname][:, tsep_kpi]
    return aftersub

def final_projection_vactype4(trajl, aftersub, otype):
    """project finally onto the operators, vactype4"""

    for num, _ in enumerate(trajl):
        for momdiag in aftersub:
            keyirr = kfp.gen_key(momdiag)
            for i in np.arange(1, 11):
                if otype == 'pipi':
                    assert str(i) in kpp.QOPI0, "Missing Q:"+str(i)
                    kpp.QOPI0[str(i)][
                        keyirr][num] += kaonprojop.qi_proj_type4(
                            aftersub[momdiag][num], i)
                elif otype == 'sigma':
                    assert str(i) in kpp.QOP_SIGMA, "Missing Q:"+str(i)
                    kpp.QOP_SIGMA[str(i)][keyirr][
                        num] += kaonprojop.qi_proj_sigma_type4(
                            aftersub[momdiag][num], i)
                else:
                    assert None, "bad otype"
