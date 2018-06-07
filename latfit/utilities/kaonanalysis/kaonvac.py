"""Vacuum subtract disconnected k->x diagrams."""

import import kaonfileproc as kfp
import kaonpostproc as kpp
import h5jack
import kaonprojop

def vacSubtractMix4(mix4dict, sinkbubbles, sinksub, trajl):
    """Vacuum subtract type4"""

    sinksub = bubsub(sinkbubbles)
    
    # jackknife type 4

    aftersub = {}
    for momdiag in mix4dict:
        momdiag = momdiag+'@000' # for backwards compatibility, means key@ptotal, ptotal=000 since Kaon is at rest
        aftersub[momdiag] = np.zeros((ltraj, 2, LT_CHECK), dtype=np.complex)
        for fidx in range(2):
            for tdis in range(LT_CHECK):
                temp_dict = {}
                temp_dict[momdiag] = type4dict[mom_diag][:, fidx, tdis, :]
                srcsub = h5jack.bubsub(temp_dict)

                # dict of averaged bubbles, to subtract
                subdict = {**srcsub, **sinksub}

                # dict of uncomposed bubbles
                bubbles = {**temp_dict, **pipibubs}

                # do the vac subtraction, avg over tk
                aftersub[momdiag][:, fidx, tdis] = h5jack.dobubjack(bubbles, subdict)

    return aftersub
        


def vacSubtractType4(type4dict, sinkbubbles, sinksub, trajl, otype):
    """Vacuum subtract type4"""

    aftersub = {}
    for momdiag in type4dict:
        momdiag = momdiag+'@000' # for backwards compatibility, means key@ptotal, ptotal=000 since Kaon is at rest
        aftersub[momdiag] = np.zeros((ltraj, 8, 4, LT_CHECK), dtype=np.complex)
        for conidx in range(8):
            for gcombidx in range(4):
                for tdis in range(LT_CHECK):
                    temp_dict = {}
                    temp_dict[momdiag] = type4dict[mom_diag][:, conidx, gcombidx, tdis, :]
                    srcsub = h5jack.bubsub(temp_dict)

                    # dict of averaged bubbles, to subtract
                    subdict = {**srcsub, **sinksub}

                    # dict of uncomposed bubbles
                    bubbles = {**temp_dict, **pipibubs}

                    # do the vac subtraction, avg over tk
                    aftersub[momdiag][:, conidx, gcombidx, tdis] = dobubjack(bubbles, subdict)

    # project finally onto the operators

    for num, traj in enumerate(trajl):
        for momdiag in aftersub:
            keyirr = kfp.genKey(momdiag)
            for i in np.arange(1, 11):
                if otype == 'pipi':
                    kpp.QOPI0[str(i)][keyirr][num] += kaonprojop.QiprojType4(aftersub[momdiag], i, 'I0')
                elif otype == 'sigma':
                    kpp.QOP_sigma[str(i)][keyirr][num] += kaonprojop.QiprojSigmaType4(aftersub[momdiag], i, 'I0')
                else:
                    assert None, "bad otype"


