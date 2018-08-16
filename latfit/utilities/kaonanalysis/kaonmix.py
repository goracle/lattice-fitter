"""Does the kaon mix3/4 uv divergence subtraction"""

import kaonpostproc as kpp
import kaonfileproc as kfp

def mixCoeffs(type4, mix4, trajl, fidx):
    """Extract the jackknifed alpha_i
    == <|Q_i|K>/<|sbar G5 d|> = <type4_bubble>/<mix4>,
    jackknife the results"""
    ltraj = len(trajl)
    alpha = {}

    # for reference
    # shapeT4 = (ltraj, 8, 4, LT_CHECK, LT_CHECK)
    # shapeM4 = (ltraj, 2, LT_CHECK, LT_CHECK)

    for momdiag in type4: # for plausibly different type4's

        alpha[momdiag] = defaultdict(lambda: np.zeros((trajl, LT_CHECK), dtype=np.complex), alpha[momdiag])

        for i in np.arange(1, 11): # assumes we've averaged over tk for type4 and mix4

            # project onto Qi
            projtemp = np.zeros((trajl, LT_CHECK), dtype=np.complex)
            for num, traj in enumerate(trajl):
                projtemp[num] = kaonprojop.QiprojType4(type4[momdiag][num], i, 'I0')

            # jackknife the results
            projtemp = dojackknife(projtemp)
            jackmix = dojackknife(mix4[momdiag][fidx])

            for num, traj in enumerate(trajl):
                alpha[momdiag][str(i)][num] = projtemp[num]/jackmix[num]
    return alpha



def mixSubtract(alpha, mix3, mix4tox, otype):
    """Subtract mix diagrams, assumes we've jackknifed the operators"""

    assert jackknifeOPS.complete, "Operators need to be jackknifed before mix subtraction."

    # mix3 = Isospin0ProjMix3(mix3) # useless
    # mix4tox = Isospin0ProjMix4(mix4tox) # useless

    # shape == key, ltraj, fidx, tdis

    for i in np.arange(1, 11):

        if otype == 'sigma':
            for momdiag in mix3:
                kpp.QOP_sigma[str(i)][kfp.genKey(momdiag)] -= alpha[momdiag][str(i)]*mix3[momdiag][:, 0, :] # 0 for pseudoscalar vertex
            for momdiag in mix4tox:
                kpp.QOP_sigma[str(i)][kfp.genKey(momdiag)] -= alpha[momdiag][str(i)]*mix4tox[momdiag][:, 0, :]

        elif otype == 'pipi':
            for momdiag in mix3:
                kpp.QOPI0[str(i)][kfp.genKey(momdiag)] -= alpha[momdiag][str(i)]*mix3[momdiag][:, 0, :] # 0 for pseudoscalar vertex
            for momdiag in mix4tox:
                kpp.QOPI0[str(i)][kfp.genKey(momdiag)] -= alpha[momdiag][str(i)]*mix4tox[momdiag][:, 0, :]

        else:
            assert None, "k->pi not written yet"

